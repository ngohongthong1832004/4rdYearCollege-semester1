import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from collections import OrderedDict

from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from peft import PeftModel
from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer

# =====================================================
# 1. CUSTOM LAYERS – GIỐNG HỆT FILE TRAIN
# =====================================================

class CustomMultiHeadAttention(nn.Module):
    """Multi-Head Attention với Scaled Dot-Product"""
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        assert self.head_dim * num_heads == hidden_size

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_length, _ = hidden_states.size()

        Q = self.query(hidden_states)
        K = self.key(hidden_states)
        V = self.value(hidden_states)

        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if attention_mask is not None:
            # attention_mask sẽ dạng (batch, seq_len)
            # ta convert sang (batch, 1, 1, seq_len) giống training
            if attention_mask.dim() == 2:
                mask = attention_mask.unsqueeze(1).unsqueeze(2)
            else:
                mask = attention_mask
            scores = scores.masked_fill(mask == 0, -1e4)

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_length, self.hidden_size
        )
        output = self.out(context)

        return output, attn_weights


class BiLSTMLayer(nn.Module):
    """Bidirectional LSTM để capture context hai chiều"""
    def __init__(self, hidden_size, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states):
        lstm_out, _ = self.lstm(hidden_states)
        output = self.layer_norm(hidden_states + self.dropout(lstm_out))
        return output


class RNNLayer(nn.Module):
    """Simple RNN layer"""
    def __init__(self, hidden_size, num_layers=1, dropout=0.1):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states):
        rnn_out, _ = self.rnn(hidden_states)
        output = self.layer_norm(hidden_states + self.dropout(rnn_out))
        return output


class ContextAttention(nn.Module):
    """Context Attention mechanism đơn giản"""
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, hidden_states):
        attention_weights = self.attention(hidden_states)
        attention_weights = self.softmax(attention_weights)
        context = torch.sum(hidden_states * attention_weights, dim=1)
        return context, attention_weights


class FeedForwardNetwork(nn.Module):
    """Position-wise Feed Forward Network"""
    def __init__(self, hidden_size, ff_size=None, dropout=0.1):
        super().__init__()
        ff_size = ff_size or hidden_size * 4

        self.fc1 = nn.Linear(hidden_size, ff_size)
        self.fc2 = nn.Linear(ff_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states):
        residual = hidden_states
        x = F.gelu(self.fc1(hidden_states))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        output = self.layer_norm(residual + x)
        return output


# =====================================================
# 2. FULL ADVANCED ARCHITECTURE (COPY TỪ TRAIN)
# =====================================================
class T5WithAdvancedArchitecture(nn.Module):
    """
    T5 Model kết hợp:
    - Multi-Head Attention (Transformer)
    - Bidirectional LSTM
    - RNN
    - Context Attention
    - Feed Forward Network
    """
    def __init__(self, base_model, hidden_size=768, num_heads=12):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config

        # 1. Multi-Head Attention
        self.multi_head_attention = CustomMultiHeadAttention(
            hidden_size, num_heads, dropout=0.1
        )

        # 2. BiLSTM
        self.bilstm = BiLSTMLayer(hidden_size, num_layers=2, dropout=0.1)

        # 3. RNN
        self.rnn = RNNLayer(hidden_size, num_layers=1, dropout=0.1)

        # 4. Context Attention (dù chưa dùng trong forward, vẫn load cho đúng weight)
        self.context_attention = ContextAttention(hidden_size)

        # 5. FFN
        self.ffn = FeedForwardNetwork(hidden_size, dropout=0.1)

        # Layer norms
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.layer_norm3 = nn.LayerNorm(hidden_size)

        # Dropout
        self.dropout = nn.Dropout(0.1)

        # Gating
        self.gate = nn.Linear(hidden_size * 3, 3)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        **kwargs,
    ):
        if "num_items_in_batch" in kwargs:
            kwargs.pop("num_items_in_batch")

        if decoder_input_ids is None and labels is not None:
            decoder_input_ids = self.base_model._shift_right(labels)

        # Encoder
        encoder_outputs = self.base_model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        encoder_hidden = encoder_outputs.last_hidden_state

        # Advanced layers
        attn_output, _ = self.multi_head_attention(
            encoder_hidden,
            attention_mask=attention_mask,
        )
        attn_output = self.layer_norm1(encoder_hidden + self.dropout(attn_output))

        lstm_output = self.bilstm(encoder_hidden)
        rnn_output = self.rnn(encoder_hidden)

        combined = torch.cat([attn_output, lstm_output, rnn_output], dim=-1)
        gates = torch.softmax(self.gate(combined), dim=-1)

        ensemble_output = (
            gates[:, :, 0:1] * attn_output
            + gates[:, :, 1:2] * lstm_output
            + gates[:, :, 2:3] * rnn_output
        )

        ensemble_output = self.ffn(ensemble_output)
        final_output = self.layer_norm3(encoder_hidden + ensemble_output)

        new_encoder_outputs = BaseModelOutput(
            last_hidden_state=final_output,
            hidden_states=encoder_outputs.hidden_states,
        )

        outputs = self.base_model(
            encoder_outputs=new_encoder_outputs,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            return_dict=True,
        )
        return outputs

    def generate(self, *args, **kwargs):
        return self.base_model.generate(*args, **kwargs)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.base_model.prepare_inputs_for_generation(*args, **kwargs)


# =====================================================
# 3. LOAD MODEL + LORA + ADVANCED LAYERS
# =====================================================
def load_model(checkpoint_dir: str):
    print(f"\n🔍 Loading checkpoint from: {checkpoint_dir}")

    # Tokenizer từ checkpoint (có thêm special tokens)
    tokenizer = T5Tokenizer.from_pretrained(checkpoint_dir)

    # Base model pretrain
    base = T5ForConditionalGeneration.from_pretrained("VietAI/vit5-base")

    # Resize embedding theo tokenizer đã lưu
    vocab_size = len(tokenizer)
    base.resize_token_embeddings(vocab_size)

    # Load LoRA adapter
    lora_path = os.path.join(checkpoint_dir, "lora_adapter")
    base = PeftModel.from_pretrained(base, lora_path)

    # Wrap advanced architecture
    hidden_size = base.config.d_model
    num_heads = base.config.num_heads
    model = T5WithAdvancedArchitecture(base_model=base,
                                       hidden_size=hidden_size,
                                       num_heads=num_heads)

    # Load advanced layers
    adv_path = os.path.join(checkpoint_dir, "advanced_layers.pt")
    sd = torch.load(adv_path, map_location="cpu")

    model.multi_head_attention.load_state_dict(sd["multi_head_attention"])
    model.bilstm.load_state_dict(sd["bilstm"])
    model.rnn.load_state_dict(sd["rnn"])
    model.context_attention.load_state_dict(sd["context_attention"])
    model.ffn.load_state_dict(sd["ffn"])
    model.gate.load_state_dict(sd["gate"])

    print("✅ Loaded LoRA + advanced_layers")

    return tokenizer, model


# =====================================================
# 4. METRICS
# =====================================================
def compute_bleu(preds, refs):
    refs_tok = [[r.split()] for r in refs]
    preds_tok = [p.split() for p in preds]
    return corpus_bleu(refs_tok, preds_tok)


def compute_rouge(preds, refs):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    r1, rl = [], []
    for p, r in zip(preds, refs):
        s = scorer.score(r, p)
        r1.append(s["rouge1"].fmeasure)
        rl.append(s["rougeL"].fmeasure)
    return float(np.mean(r1)), float(np.mean(rl))


# =====================================================
# 5. MODEL ANALYSIS FUNCTIONS (CHO BÁO CÁO)
# =====================================================

def count_parameters(model):
    """Đếm số lượng parameters của model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'frozen_params': frozen_params,
        'total_params_M': total_params / 1e6,
        'trainable_params_M': trainable_params / 1e6,
    }


def get_model_size(model):
    """Tính kích thước model (MB)"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def measure_inference_time(model, tokenizer, test_texts, device, num_runs=100):
    """Đo thời gian inference và tính FPS"""
    model.eval()
    times = []
    
    print(f"\n⏱️  Đang đo inference time với {num_runs} samples...")
    
    with torch.no_grad():
        # Warmup
        for i in range(min(10, len(test_texts))):
            inputs = tokenizer.encode(test_texts[i], return_tensors="pt").to(device)
            _ = model.generate(input_ids=inputs, max_length=64, num_beams=4)
        
        # Actual measurement
        for i in range(min(num_runs, len(test_texts))):
            text = test_texts[i]
            inputs = tokenizer.encode(text, return_tensors="pt").to(device)
            
            start_time = time.time()
            _ = model.generate(
                input_ids=inputs,
                max_length=64,
                num_beams=4,
                early_stopping=True
            )
            end_time = time.time()
            
            times.append(end_time - start_time)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    fps = 1.0 / avg_time if avg_time > 0 else 0
    
    return {
        'avg_inference_time': avg_time,
        'std_inference_time': std_time,
        'min_inference_time': np.min(times),
        'max_inference_time': np.max(times),
        'fps': fps,
        'samples_per_second': fps,
    }


def get_model_architecture_info(model):
    """Lấy thông tin kiến trúc model"""
    info = OrderedDict()
    
    # Unwrap nếu có DataParallel
    model_to_check = model.module if hasattr(model, 'module') else model
    
    info['model_type'] = 'T5WithAdvancedArchitecture'
    info['base_model'] = 'VietAI/vit5-base'
    
    # Kiểm tra các components
    info['has_multi_head_attention'] = hasattr(model_to_check, 'multi_head_attention')
    info['has_bilstm'] = hasattr(model_to_check, 'bilstm')
    info['has_rnn'] = hasattr(model_to_check, 'rnn')
    info['has_context_attention'] = hasattr(model_to_check, 'context_attention')
    info['has_ffn'] = hasattr(model_to_check, 'ffn')
    info['has_lora'] = hasattr(model_to_check, 'base_model') and hasattr(model_to_check.base_model, 'peft_config')
    
    # Lấy config từ base model
    if hasattr(model_to_check, 'config'):
        config = model_to_check.config
        info['hidden_size'] = config.d_model if hasattr(config, 'd_model') else 'N/A'
        info['num_layers'] = config.num_layers if hasattr(config, 'num_layers') else 'N/A'
        info['num_heads'] = config.num_heads if hasattr(config, 'num_heads') else 'N/A'
        info['vocab_size'] = config.vocab_size if hasattr(config, 'vocab_size') else 'N/A'
    
    return info


def get_gpu_memory_usage():
    """Lấy thông tin sử dụng GPU memory"""
    if not torch.cuda.is_available():
        return None
    
    memory_info = {}
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        
        memory_info[f'gpu_{i}'] = {
            'name': torch.cuda.get_device_name(i),
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'total_gb': total,
            'utilization_percent': (allocated / total) * 100 if total > 0 else 0,
        }
    
    return memory_info


def print_model_report(model, tokenizer, test_texts, device):
    """In báo cáo tổng hợp về model"""
    print("\n" + "="*80)
    print("📊 BÁO CÁO CHI TIẾT MÔ HÌNH")
    print("="*80)
    
    # 1. Thông số parameters
    print("\n1️⃣  THÔNG SỐ PARAMETERS:")
    print("-" * 80)
    param_info = count_parameters(model)
    print(f"   • Tổng parameters:      {param_info['total_params']:,} ({param_info['total_params_M']:.2f}M)")
    print(f"   • Trainable parameters: {param_info['trainable_params']:,} ({param_info['trainable_params_M']:.2f}M)")
    print(f"   • Frozen parameters:    {param_info['frozen_params']:,}")
    print(f"   • Tỷ lệ trainable:      {(param_info['trainable_params']/param_info['total_params']*100):.2f}%")
    
    # 2. Kích thước model
    print("\n2️⃣  KÍCH THƯỚC MÔ HÌNH:")
    print("-" * 80)
    model_size = get_model_size(model)
    print(f"   • Model size:           {model_size:.2f} MB")
    print(f"   • Tokenizer vocab size: {len(tokenizer):,}")
    
    # 3. Kiến trúc
    print("\n3️⃣  KIẾN TRÚC MÔ HÌNH:")
    print("-" * 80)
    arch_info = get_model_architecture_info(model)
    for key, value in arch_info.items():
        print(f"   • {key:.<30} {value}")
    
    # 4. Inference performance
    print("\n4️⃣  HIỆU SUẤT INFERENCE:")
    print("-" * 80)
    perf_info = measure_inference_time(model, tokenizer, test_texts, device)
    print(f"   • Avg inference time:   {perf_info['avg_inference_time']*1000:.2f} ms")
    print(f"   • Std inference time:   {perf_info['std_inference_time']*1000:.2f} ms")
    print(f"   • Min inference time:   {perf_info['min_inference_time']*1000:.2f} ms")
    print(f"   • Max inference time:   {perf_info['max_inference_time']*1000:.2f} ms")
    print(f"   • FPS (samples/sec):    {perf_info['fps']:.2f}")
    
    # 5. GPU Memory
    print("\n5️⃣  SỬ DỤNG GPU MEMORY:")
    print("-" * 80)
    gpu_info = get_gpu_memory_usage()
    if gpu_info:
        for gpu_name, info in gpu_info.items():
            print(f"   • {info['name']}:")
            print(f"     - Allocated: {info['allocated_gb']:.2f} GB")
            print(f"     - Reserved:  {info['reserved_gb']:.2f} GB")
            print(f"     - Total:     {info['total_gb']:.2f} GB")
            print(f"     - Usage:     {info['utilization_percent']:.1f}%")
    else:
        print("   • No GPU available")
    
    print("\n" + "="*80)
    
    return {
        'parameters': param_info,
        'model_size_mb': model_size,
        'architecture': arch_info,
        'performance': perf_info,
        'gpu_memory': gpu_info,
    }


def save_model_report_to_file(report_data, output_file="model_report.txt"):
    """Lưu báo cáo ra file text"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("📊 BÁO CÁO CHI TIẾT MÔ HÌNH\n")
        f.write("="*80 + "\n\n")
        
        f.write("1️⃣  THÔNG SỐ PARAMETERS:\n")
        f.write("-" * 80 + "\n")
        param_info = report_data['parameters']
        f.write(f"Tổng parameters:      {param_info['total_params']:,} ({param_info['total_params_M']:.2f}M)\n")
        f.write(f"Trainable parameters: {param_info['trainable_params']:,} ({param_info['trainable_params_M']:.2f}M)\n")
        f.write(f"Frozen parameters:    {param_info['frozen_params']:,}\n")
        f.write(f"Tỷ lệ trainable:      {(param_info['trainable_params']/param_info['total_params']*100):.2f}%\n\n")
        
        f.write("2️⃣  KÍCH THƯỚC MÔ HÌNH:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Model size: {report_data['model_size_mb']:.2f} MB\n\n")
        
        f.write("3️⃣  KIẾN TRÚC MÔ HÌNH:\n")
        f.write("-" * 80 + "\n")
        for key, value in report_data['architecture'].items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        f.write("4️⃣  HIỆU SUẤT INFERENCE:\n")
        f.write("-" * 80 + "\n")
        perf = report_data['performance']
        f.write(f"Avg inference time: {perf['avg_inference_time']*1000:.2f} ms\n")
        f.write(f"Std inference time: {perf['std_inference_time']*1000:.2f} ms\n")
        f.write(f"FPS (samples/sec):  {perf['fps']:.2f}\n\n")
        
        if report_data['gpu_memory']:
            f.write("5️⃣  SỬ DỤNG GPU MEMORY:\n")
            f.write("-" * 80 + "\n")
            for gpu_name, info in report_data['gpu_memory'].items():
                f.write(f"{info['name']}:\n")
                f.write(f"  Allocated: {info['allocated_gb']:.2f} GB\n")
                f.write(f"  Total:     {info['total_gb']:.2f} GB\n")
                f.write(f"  Usage:     {info['utilization_percent']:.1f}%\n")
    
    print(f"💾 Saved detailed report to: {output_file}")


# =====================================================
# 6. EVALUATE + SAVE CSV + CHART
# =====================================================
def evaluate_model(checkpoint_dir, test_csv):
    tokenizer, model = load_model(checkpoint_dir)

    df = pd.read_csv(test_csv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    preds = []
    refs = []
    test_texts = []

    print("\n🚀 Generating predictions...")
    for _, row in df.iterrows():
        text_in = row["input_text"]
        text_ref = row["target_text"]
        test_texts.append(text_in)

        inputs = tokenizer.encode(text_in, return_tensors="pt").to(device)

        with torch.no_grad():
            out_ids = model.generate(
                input_ids=inputs,
                max_length=64,
                num_beams=4,
                early_stopping=True
            )
        pred = tokenizer.decode(out_ids[0], skip_special_tokens=True)

        preds.append(pred)
        refs.append(text_ref)

    bleu = compute_bleu(preds, refs)
    r1, rl = compute_rouge(preds, refs)

    print("\n====== 📊 EVALUATION METRICS ======")
    print(f"BLEU      : {bleu:.4f}")
    print(f"ROUGE-1 F1: {r1:.4f}")
    print(f"ROUGE-L F1: {rl:.4f}")

    # Lưu CSV kết quả
    out_df = pd.DataFrame({
        "input_text": df["input_text"],
        "target_text": refs,
        "pred_text": preds
    })
    out_df.to_csv("eval_results.csv", index=False)
    print("💾 Saved eval_results.csv")

    # Vẽ chart metrics
    plt.figure(figsize=(8,5))
    metrics = ["BLEU", "ROUGE-1", "ROUGE-L"]
    values = [bleu, r1, rl]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = plt.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Thêm giá trị trên mỗi cột
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.ylim(0, 1)
    plt.ylabel("Score", fontsize=12, fontweight='bold')
    plt.title("Evaluation Metrics", fontsize=14, fontweight='bold')
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig("metrics_chart.png", dpi=300, bbox_inches='tight')
    print("📊 Saved metrics_chart.png")
    
    # In báo cáo chi tiết model
    report_data = print_model_report(model, tokenizer, test_texts, device)
    
    # Lưu báo cáo ra file
    save_model_report_to_file(report_data, "model_report.txt")
    
    # Vẽ thêm chart so sánh parameters
    plt.figure(figsize=(10, 6))
    param_info = report_data['parameters']
    
    plt.subplot(1, 2, 1)
    sizes = [param_info['trainable_params_M'], param_info['total_params_M'] - param_info['trainable_params_M']]
    labels = ['Trainable', 'Frozen']
    colors_pie = ['#4ECDC4', '#95E1D3']
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors_pie, startangle=90)
    plt.title('Parameters Distribution', fontweight='bold')
    
    plt.subplot(1, 2, 2)
    perf_info = report_data['performance']
    perf_metrics = ['Avg Time\n(ms)', 'Min Time\n(ms)', 'Max Time\n(ms)']
    perf_values = [
        perf_info['avg_inference_time'] * 1000,
        perf_info['min_inference_time'] * 1000,
        perf_info['max_inference_time'] * 1000
    ]
    bars = plt.bar(perf_metrics, perf_values, color=['#FF6B6B', '#95E1D3', '#FFE66D'], alpha=0.8, edgecolor='black')
    
    for bar, value in zip(bars, perf_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.1f}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.ylabel('Time (ms)', fontweight='bold')
    plt.title('Inference Performance', fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("model_analysis.png", dpi=300, bbox_inches='tight')
    print("📊 Saved model_analysis.png")

    return bleu, r1, rl


# =====================================================
# 7. RUN
# =====================================================
if __name__ == "__main__":
    CHECKPOINT_DIR = "../weight/checkpoint-4400"   # chỉnh nếu path khác
    TEST_CSV = "test.csv"         # chỉnh nếu path khác

    print("="*80)
    print("🚀 ĐÁNH GIÁ MÔ HÌNH VÀ TẠO BÁO CÁO CHI TIẾT")
    print("="*80)
    
    evaluate_model(CHECKPOINT_DIR, TEST_CSV)
    
    print("\n" + "="*80)
    print("✅ HOÀN TẤT! Đã tạo các file:")
    print("   📄 eval_results.csv        - Kết quả dự đoán chi tiết")
    print("   📊 metrics_chart.png       - Biểu đồ các metrics (BLEU, ROUGE)")
    print("   📊 model_analysis.png      - Phân tích parameters và performance")
    print("   📝 model_report.txt        - Báo cáo chi tiết về model")
    print("="*80)

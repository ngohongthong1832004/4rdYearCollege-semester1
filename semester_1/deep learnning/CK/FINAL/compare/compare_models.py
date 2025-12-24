"""
Script so sánh chi tiết giữa MT5 và ViT5 models
Tạo bảng và biểu đồ so sánh toàn diện
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
import math

# Import từ transformers
from transformers import (
    T5Tokenizer, 
    T5ForConditionalGeneration,
    MT5ForConditionalGeneration,
    AutoTokenizer
)
from transformers.modeling_outputs import BaseModelOutput
from peft import PeftModel

# Import metrics
from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set style cho plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


# =====================================================
# CUSTOM LAYERS CHO VIT5 (ADVANCED ARCHITECTURE)
# =====================================================

class CustomMultiHeadAttention(nn.Module):
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
            if attention_mask.dim() == 2:
                mask = attention_mask.unsqueeze(1).unsqueeze(2)
            else:
                mask = attention_mask
            scores = scores.masked_fill(mask == 0, -1e4)
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_size)
        return self.out(context), attn_weights


class BiLSTMLayer(nn.Module):
    def __init__(self, hidden_size, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=hidden_size, hidden_size=hidden_size // 2,
            num_layers=num_layers, dropout=dropout if num_layers > 1 else 0,
            bidirectional=True, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states):
        lstm_out, _ = self.lstm(hidden_states)
        return self.layer_norm(hidden_states + self.dropout(lstm_out))


class RNNLayer(nn.Module):
    def __init__(self, hidden_size, num_layers=1, dropout=0.1):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=hidden_size, hidden_size=hidden_size,
            num_layers=num_layers, dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states):
        rnn_out, _ = self.rnn(hidden_states)
        return self.layer_norm(hidden_states + self.dropout(rnn_out))


class ContextAttention(nn.Module):
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
        return self.layer_norm(residual + x)


class T5WithAdvancedArchitecture(nn.Module):
    """ViT5 với Advanced Architecture"""
    def __init__(self, base_model, hidden_size=768, num_heads=12):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        
        self.multi_head_attention = CustomMultiHeadAttention(hidden_size, num_heads, dropout=0.1)
        self.bilstm = BiLSTMLayer(hidden_size, num_layers=2, dropout=0.1)
        self.rnn = RNNLayer(hidden_size, num_layers=1, dropout=0.1)
        self.context_attention = ContextAttention(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, dropout=0.1)
        
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.layer_norm3 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.gate = nn.Linear(hidden_size * 3, 3)

    def forward(self, input_ids=None, attention_mask=None, labels=None, 
                decoder_input_ids=None, decoder_attention_mask=None, **kwargs):
        if "num_items_in_batch" in kwargs:
            kwargs.pop("num_items_in_batch")
        if decoder_input_ids is None and labels is not None:
            decoder_input_ids = self.base_model._shift_right(labels)

        encoder_outputs = self.base_model.encoder(
            input_ids=input_ids, attention_mask=attention_mask,
            output_hidden_states=True, return_dict=True
        )
        encoder_hidden = encoder_outputs.last_hidden_state

        attn_output, _ = self.multi_head_attention(encoder_hidden, attention_mask=attention_mask)
        attn_output = self.layer_norm1(encoder_hidden + self.dropout(attn_output))
        lstm_output = self.bilstm(encoder_hidden)
        rnn_output = self.rnn(encoder_hidden)

        combined = torch.cat([attn_output, lstm_output, rnn_output], dim=-1)
        gates = torch.softmax(self.gate(combined), dim=-1)
        ensemble_output = (
            gates[:, :, 0:1] * attn_output +
            gates[:, :, 1:2] * lstm_output +
            gates[:, :, 2:3] * rnn_output
        )
        ensemble_output = self.ffn(ensemble_output)
        final_output = self.layer_norm3(encoder_hidden + ensemble_output)

        new_encoder_outputs = BaseModelOutput(
            last_hidden_state=final_output,
            hidden_states=encoder_outputs.hidden_states
        )

        outputs = self.base_model(
            encoder_outputs=new_encoder_outputs, attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels, return_dict=True
        )
        return outputs

    def generate(self, *args, **kwargs):
        return self.base_model.generate(*args, **kwargs)


# =====================================================
# LOAD MODELS
# =====================================================

def load_vit5_model(checkpoint_dir):
    """Load ViT5 model with advanced architecture"""
    print(f"\n🔍 Loading ViT5 from: {checkpoint_dir}")
    
    tokenizer = T5Tokenizer.from_pretrained(checkpoint_dir)
    base = T5ForConditionalGeneration.from_pretrained("VietAI/vit5-base")
    base.resize_token_embeddings(len(tokenizer))
    
    lora_path = os.path.join(checkpoint_dir, "lora_adapter")
    base = PeftModel.from_pretrained(base, lora_path)
    
    hidden_size = base.config.d_model
    num_heads = base.config.num_heads
    model = T5WithAdvancedArchitecture(base_model=base, hidden_size=hidden_size, num_heads=num_heads)
    
    adv_path = os.path.join(checkpoint_dir, "advanced_layers.pt")
    sd = torch.load(adv_path, map_location="cpu")
    model.multi_head_attention.load_state_dict(sd["multi_head_attention"])
    model.bilstm.load_state_dict(sd["bilstm"])
    model.rnn.load_state_dict(sd["rnn"])
    model.context_attention.load_state_dict(sd["context_attention"])
    model.ffn.load_state_dict(sd["ffn"])
    model.gate.load_state_dict(sd["gate"])
    
    print("✅ Loaded ViT5 + LoRA + Advanced Layers")
    return tokenizer, model


def load_mt5_model(checkpoint_dir):
    """Load MT5 model with LoRA only"""
    print(f"\n🔍 Loading MT5 from: {checkpoint_dir}")
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = MT5ForConditionalGeneration.from_pretrained(
        "google/mt5-base",
        torch_dtype=torch.bfloat16
    )
    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, checkpoint_dir)
    
    print("✅ Loaded MT5 + LoRA")
    return tokenizer, model


# =====================================================
# METRICS COMPUTATION
# =====================================================

def compute_metrics(preds, refs):
    """Tính BLEU và ROUGE scores"""
    # Filter out empty predictions
    valid_pairs = [(p, r) for p, r in zip(preds, refs) if p and p.strip()]
    
    if not valid_pairs:
        print("   ⚠️ Warning: No valid predictions found!")
        return {
            'bleu': 0.0,
            'rouge1': 0.0,
            'rouge2': 0.0,
            'rougeL': 0.0,
        }
    
    preds_valid = [p for p, r in valid_pairs]
    refs_valid = [r for p, r in valid_pairs]
    
    # BLEU with smoothing
    from nltk.translate.bleu_score import SmoothingFunction
    smoothie = SmoothingFunction().method4
    
    refs_tok = [[r.split()] for r in refs_valid]
    preds_tok = [p.split() for p in preds_valid]
    bleu = corpus_bleu(refs_tok, preds_tok, smoothing_function=smoothie)
    
    # ROUGE
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1_scores, r2_scores, rl_scores = [], [], []
    
    for p, r in zip(preds_valid, refs_valid):
        score = scorer.score(r, p)
        r1_scores.append(score["rouge1"].fmeasure)
        r2_scores.append(score["rouge2"].fmeasure)
        rl_scores.append(score["rougeL"].fmeasure)
    
    return {
        'bleu': bleu,
        'rouge1': np.mean(r1_scores),
        'rouge2': np.mean(r2_scores),
        'rougeL': np.mean(rl_scores),
    }


def count_parameters(model):
    """Đếm parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable,
        'trainable_ratio': trainable / total if total > 0 else 0
    }


def get_model_size_mb(model):
    """Tính kích thước model (MB)"""
    size = 0
    for p in model.parameters():
        size += p.nelement() * p.element_size()
    for buffer in model.buffers():
        size += buffer.nelement() * buffer.element_size()
    return size / (1024**2)


def measure_inference_speed(model, tokenizer, texts, device, model_name, runs=100):
    """Đo tốc độ inference"""
    model.eval()
    times = []
    
    with torch.no_grad():
        # Warmup
        for i in range(min(10, len(texts))):
            try:
                if model_name == "MT5":
                    inp = tokenizer(texts[i], return_tensors="pt", max_length=512, truncation=True).to(device)
                    _ = model.generate(**inp, max_length=64, num_beams=4)
                else:
                    inp_ids = tokenizer.encode(texts[i], return_tensors="pt", max_length=512, truncation=True).to(device)
                    _ = model.generate(input_ids=inp_ids, max_length=64, num_beams=4)
            except:
                pass
        
        # Actual measurement
        for i in range(min(runs, len(texts))):
            try:
                t0 = time.time()
                
                if model_name == "MT5":
                    inp = tokenizer(texts[i], return_tensors="pt", max_length=512, truncation=True).to(device)
                    _ = model.generate(**inp, max_length=64, num_beams=4)
                else:
                    inp_ids = tokenizer.encode(texts[i], return_tensors="pt", max_length=512, truncation=True).to(device)
                    _ = model.generate(input_ids=inp_ids, max_length=64, num_beams=4)
                
                t1 = time.time()
                times.append(t1 - t0)
            except:
                pass
    
    if not times:
        return {
            'avg_time': 0,
            'std_time': 0,
            'min_time': 0,
            'max_time': 0,
            'fps': 0
        }
    
    return {
        'avg_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'fps': 1.0 / np.mean(times) if np.mean(times) > 0 else 0
    }


# =====================================================
# EVALUATION
# =====================================================

def evaluate_model(model, tokenizer, test_df, device, model_name):
    """Đánh giá một model"""
    print(f"\n📊 Evaluating {model_name}...")
    
    model.to(device)
    model.eval()
    
    preds = []
    refs = []
    test_texts = []
    
    for idx, row in test_df.iterrows():
        text_in = row["input_text"]
        text_ref = row["target_text"]
        test_texts.append(text_in)
        refs.append(text_ref)
        
        # Generate prediction
        with torch.no_grad():
            try:
                if model_name == "MT5":
                    # MT5 model - use tokenizer() with dict unpacking
                    inp = tokenizer(text_in, return_tensors="pt", max_length=512, truncation=True).to(device)
                    out = model.generate(
                        **inp, 
                        max_length=64,
                        min_length=5,
                        num_beams=4,
                        early_stopping=True,
                        no_repeat_ngram_size=2
                    )
                else:
                    # ViT5 model with advanced architecture - use encode()
                    inp_ids = tokenizer.encode(text_in, return_tensors="pt", max_length=512, truncation=True).to(device)
                    out = model.generate(
                        input_ids=inp_ids,
                        max_length=64,
                        min_length=5,
                        num_beams=4,
                        early_stopping=True,
                        no_repeat_ngram_size=2
                    )
                
                pred = tokenizer.decode(out[0], skip_special_tokens=True)
                
                # Debug: print first few predictions
                if idx < 3:
                    print(f"   Sample {idx+1}: '{pred[:100]}...' " if len(pred) > 100 else f"   Sample {idx+1}: '{pred}'")
                
            except Exception as e:
                print(f"   ⚠️ Error at sample {idx}: {e}")
                pred = ""
            
            preds.append(pred)
    
    print(f"   ✅ Generated {len([p for p in preds if p])} non-empty predictions out of {len(preds)}")
    
    # Compute metrics
    metrics = compute_metrics(preds, refs)
    
    # Model info
    params = count_parameters(model)
    model_size = get_model_size_mb(model)
    speed = measure_inference_speed(model, tokenizer, test_texts, device, model_name)
    
    return {
        'model_name': model_name,
        'predictions': preds,
        'references': refs,
        'metrics': metrics,
        'parameters': params,
        'model_size_mb': model_size,
        'speed': speed,
    }


# =====================================================
# COMPARISON & VISUALIZATION
# =====================================================

def create_comparison_table(vit5_results, mt5_results):
    """Tạo bảng so sánh chi tiết"""
    
    data = {
        'Metric': [],
        'ViT5': [],
        'MT5': [],
        'Difference': [],
        'Winner': []
    }
    
    # Performance metrics
    metrics_to_compare = [
        ('BLEU', 'metrics', 'bleu'),
        ('ROUGE-1', 'metrics', 'rouge1'),
        ('ROUGE-2', 'metrics', 'rouge2'),
        ('ROUGE-L', 'metrics', 'rougeL'),
    ]
    
    for metric_name, category, key in metrics_to_compare:
        vit5_val = vit5_results[category][key]
        mt5_val = mt5_results[category][key]
        diff = vit5_val - mt5_val
        winner = 'ViT5' if vit5_val > mt5_val else 'MT5' if mt5_val > vit5_val else 'Tie'
        
        data['Metric'].append(metric_name)
        data['ViT5'].append(f"{vit5_val:.4f}")
        data['MT5'].append(f"{mt5_val:.4f}")
        data['Difference'].append(f"{diff:+.4f}")
        data['Winner'].append(winner)
    
    # Model characteristics
    data['Metric'].extend([
        'Total Parameters (M)',
        'Trainable Params (M)',
        'Model Size (MB)',
        'Avg Inference Time (ms)',
        'FPS (samples/sec)'
    ])
    
    vit5_params = vit5_results['parameters']['total'] / 1e6
    mt5_params = mt5_results['parameters']['total'] / 1e6
    data['ViT5'].append(f"{vit5_params:.2f}")
    data['MT5'].append(f"{mt5_params:.2f}")
    data['Difference'].append(f"{vit5_params - mt5_params:+.2f}")
    data['Winner'].append('MT5' if mt5_params < vit5_params else 'ViT5')
    
    vit5_train = vit5_results['parameters']['trainable'] / 1e6
    mt5_train = mt5_results['parameters']['trainable'] / 1e6
    data['ViT5'].append(f"{vit5_train:.2f}")
    data['MT5'].append(f"{mt5_train:.2f}")
    data['Difference'].append(f"{vit5_train - mt5_train:+.2f}")
    data['Winner'].append('MT5' if mt5_train < vit5_train else 'ViT5')
    
    vit5_size = vit5_results['model_size_mb']
    mt5_size = mt5_results['model_size_mb']
    data['ViT5'].append(f"{vit5_size:.2f}")
    data['MT5'].append(f"{mt5_size:.2f}")
    data['Difference'].append(f"{vit5_size - mt5_size:+.2f}")
    data['Winner'].append('MT5' if mt5_size < vit5_size else 'ViT5')
    
    vit5_time = vit5_results['speed']['avg_time'] * 1000
    mt5_time = mt5_results['speed']['avg_time'] * 1000
    data['ViT5'].append(f"{vit5_time:.2f}")
    data['MT5'].append(f"{mt5_time:.2f}")
    data['Difference'].append(f"{vit5_time - mt5_time:+.2f}")
    data['Winner'].append('MT5' if mt5_time < vit5_time else 'ViT5')
    
    vit5_fps = vit5_results['speed']['fps']
    mt5_fps = mt5_results['speed']['fps']
    data['ViT5'].append(f"{vit5_fps:.2f}")
    data['MT5'].append(f"{mt5_fps:.2f}")
    data['Difference'].append(f"{vit5_fps - mt5_fps:+.2f}")
    data['Winner'].append('ViT5' if vit5_fps > mt5_fps else 'MT5')
    
    df = pd.DataFrame(data)
    return df


def plot_metrics_comparison(vit5_results, mt5_results, output_dir="."):
    """Vẽ biểu đồ so sánh metrics"""
    
    # 1. Performance Metrics Comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Comparison: ViT5 vs MT5', fontsize=16, fontweight='bold')
    
    # BLEU & ROUGE scores
    metrics_names = ['BLEU', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L']
    vit5_metrics = [
        vit5_results['metrics']['bleu'],
        vit5_results['metrics']['rouge1'],
        vit5_results['metrics']['rouge2'],
        vit5_results['metrics']['rougeL']
    ]
    mt5_metrics = [
        mt5_results['metrics']['bleu'],
        mt5_results['metrics']['rouge1'],
        mt5_results['metrics']['rouge2'],
        mt5_results['metrics']['rougeL']
    ]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    ax = axes[0, 0]
    bars1 = ax.bar(x - width/2, vit5_metrics, width, label='ViT5', color='#FF6B6B', alpha=0.8)
    bars2 = ax.bar(x + width/2, mt5_metrics, width, label='MT5', color='#4ECDC4', alpha=0.8)
    
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Performance Metrics', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 2. Model Size Comparison
    ax = axes[0, 1]
    sizes = ['Total Params\n(M)', 'Trainable\n(M)', 'Model Size\n(MB)']
    vit5_sizes = [
        vit5_results['parameters']['total'] / 1e6,
        vit5_results['parameters']['trainable'] / 1e6,
        vit5_results['model_size_mb']
    ]
    mt5_sizes = [
        mt5_results['parameters']['total'] / 1e6,
        mt5_results['parameters']['trainable'] / 1e6,
        mt5_results['model_size_mb']
    ]
    
    x = np.arange(len(sizes))
    bars1 = ax.bar(x - width/2, vit5_sizes, width, label='ViT5', color='#FF6B6B', alpha=0.8)
    bars2 = ax.bar(x + width/2, mt5_sizes, width, label='MT5', color='#4ECDC4', alpha=0.8)
    
    ax.set_ylabel('Value', fontweight='bold')
    ax.set_title('Model Size Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    # 3. Inference Speed Comparison
    ax = axes[1, 0]
    speed_metrics = ['Avg Time\n(ms)', 'Min Time\n(ms)', 'Max Time\n(ms)']
    vit5_speeds = [
        vit5_results['speed']['avg_time'] * 1000,
        vit5_results['speed']['min_time'] * 1000,
        vit5_results['speed']['max_time'] * 1000
    ]
    mt5_speeds = [
        mt5_results['speed']['avg_time'] * 1000,
        mt5_results['speed']['min_time'] * 1000,
        mt5_results['speed']['max_time'] * 1000
    ]
    
    x = np.arange(len(speed_metrics))
    bars1 = ax.bar(x - width/2, vit5_speeds, width, label='ViT5', color='#FF6B6B', alpha=0.8)
    bars2 = ax.bar(x + width/2, mt5_speeds, width, label='MT5', color='#4ECDC4', alpha=0.8)
    
    ax.set_ylabel('Time (ms)', fontweight='bold')
    ax.set_title('Inference Speed', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(speed_metrics)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    # 4. FPS Comparison (Throughput)
    ax = axes[1, 1]
    models = ['ViT5', 'MT5']
    fps_values = [vit5_results['speed']['fps'], mt5_results['speed']['fps']]
    colors_fps = ['#FF6B6B', '#4ECDC4']
    
    bars = ax.bar(models, fps_values, color=colors_fps, alpha=0.8, edgecolor='black')
    ax.set_ylabel('Samples/Second', fontweight='bold')
    ax.set_title('Throughput (FPS)', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, fps_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_comparison.png", dpi=300, bbox_inches='tight')
    print(f"📊 Saved: {output_dir}/model_comparison.png")
    plt.close()


def plot_detailed_analysis(vit5_results, mt5_results, output_dir="."):
    """Vẽ biểu đồ phân tích chi tiết"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Detailed Analysis: ViT5 vs MT5', fontsize=16, fontweight='bold')
    
    # 1. Win/Loss Chart
    ax = axes[0]
    comparison_df = create_comparison_table(vit5_results, mt5_results)
    vit5_wins = (comparison_df['Winner'] == 'ViT5').sum()
    mt5_wins = (comparison_df['Winner'] == 'MT5').sum()
    ties = (comparison_df['Winner'] == 'Tie').sum()
    
    wedges, texts, autotexts = ax.pie(
        [vit5_wins, mt5_wins, ties],
        labels=['ViT5 Wins', 'MT5 Wins', 'Ties'],
        colors=['#FF6B6B', '#4ECDC4', '#95E1D3'],
        autopct='%1.1f%%',
        startangle=90
    )
    ax.set_title('Overall Winner', fontweight='bold')
    
    # 2. Parameters Distribution
    ax = axes[1]
    vit5_trainable = vit5_results['parameters']['trainable'] / 1e6
    vit5_frozen = vit5_results['parameters']['frozen'] / 1e6
    mt5_trainable = mt5_results['parameters']['trainable'] / 1e6
    mt5_frozen = mt5_results['parameters']['frozen'] / 1e6
    
    width = 0.35
    x = np.array([0, 1])
    
    p1 = ax.bar(x - width/2, [vit5_trainable, mt5_trainable], width, 
                label='Trainable', color='#4ECDC4', alpha=0.8)
    p2 = ax.bar(x - width/2, [vit5_frozen, mt5_frozen], width, bottom=[vit5_trainable, mt5_trainable],
                label='Frozen', color='#95E1D3', alpha=0.8)
    
    ax.set_ylabel('Parameters (M)', fontweight='bold')
    ax.set_title('Parameters Distribution', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['ViT5', 'MT5'])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 3. Efficiency Score (Custom metric)
    ax = axes[2]
    
    # Calculate efficiency: (BLEU score) / (inference time)
    vit5_efficiency = vit5_results['metrics']['bleu'] / (vit5_results['speed']['avg_time'] * 1000)
    mt5_efficiency = mt5_results['metrics']['bleu'] / (mt5_results['speed']['avg_time'] * 1000)
    
    models = ['ViT5', 'MT5']
    efficiency = [vit5_efficiency, mt5_efficiency]
    colors = ['#FF6B6B', '#4ECDC4']
    
    bars = ax.bar(models, efficiency, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('Efficiency (BLEU/ms)', fontweight='bold')
    ax.set_title('Model Efficiency', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, efficiency):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/detailed_analysis.png", dpi=300, bbox_inches='tight')
    print(f"📊 Saved: {output_dir}/detailed_analysis.png")
    plt.close()


# =====================================================
# MAIN COMPARISON
# =====================================================

def compare_models(vit5_checkpoint, mt5_checkpoint, test_csv, output_dir="."):
    """So sánh toàn diện giữa ViT5 và MT5"""
    
    print("="*80)
    print("🔬 MODEL COMPARISON: ViT5 vs MT5")
    print("="*80)
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n💻 Device: {device}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load test data
    print(f"\n📂 Loading test data from: {test_csv}")
    test_df = pd.read_csv(test_csv)
    print(f"   Test samples: {len(test_df)}")
    
    # Load models
    vit5_tokenizer, vit5_model = load_vit5_model(vit5_checkpoint)
    mt5_tokenizer, mt5_model = load_mt5_model(mt5_checkpoint)
    
    # Evaluate both models
    vit5_results = evaluate_model(vit5_model, vit5_tokenizer, test_df, device, "ViT5")
    mt5_results = evaluate_model(mt5_model, mt5_tokenizer, test_df, device, "MT5")
    
    # Create comparison table
    print("\n" + "="*80)
    print("📊 COMPARISON TABLE")
    print("="*80)
    comparison_df = create_comparison_table(vit5_results, mt5_results)
    print(comparison_df.to_string(index=False))
    
    # Save comparison table
    comparison_df.to_csv(f"{output_dir}/comparison_table.csv", index=False)
    print(f"\n💾 Saved: {output_dir}/comparison_table.csv")
    
    # Save detailed results
    pd.DataFrame({
        'input_text': test_df['input_text'],
        'target_text': vit5_results['references'],
        'vit5_prediction': vit5_results['predictions'],
        'mt5_prediction': mt5_results['predictions']
    }).to_csv(f"{output_dir}/predictions_comparison.csv", index=False)
    print(f"💾 Saved: {output_dir}/predictions_comparison.csv")
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    plot_metrics_comparison(vit5_results, mt5_results, output_dir)
    plot_detailed_analysis(vit5_results, mt5_results, output_dir)
    
    # Summary
    print("\n" + "="*80)
    print("✅ COMPARISON COMPLETE!")
    print("="*80)
    print(f"\n📁 Output files saved in: {output_dir}/")
    print("   📄 comparison_table.csv         - Bảng so sánh chi tiết")
    print("   📄 predictions_comparison.csv   - So sánh predictions")
    print("   📊 model_comparison.png         - Biểu đồ so sánh tổng quan")
    print("   📊 detailed_analysis.png        - Phân tích chi tiết")
    print("="*80)
    
    return vit5_results, mt5_results, comparison_df


# =====================================================
# RUN
# =====================================================

if __name__ == "__main__":
    # Paths
    VIT5_CHECKPOINT = "../vit5/weight/checkpoint-4400"
    MT5_CHECKPOINT = "../mt5/weight/checkpoint-300"
    TEST_CSV = "../vit5/code/test.csv"  # hoặc ../mt5/code/test.csv (nên giống nhau)
    OUTPUT_DIR = "./comparison_results"
    
    # Run comparison
    vit5_results, mt5_results, comparison_df = compare_models(
        vit5_checkpoint=VIT5_CHECKPOINT,
        mt5_checkpoint=MT5_CHECKPOINT,
        test_csv=TEST_CSV,
        output_dir=OUTPUT_DIR
    )

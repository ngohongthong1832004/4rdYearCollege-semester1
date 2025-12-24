import os
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict

from transformers import (
    MT5ForConditionalGeneration,
    AutoTokenizer
)
from peft import PeftModel

from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# =====================================================
# 1. LOAD MODEL (mT5 + LoRA ONLY - Khớp với training)
# =====================================================
def load_model_mt5(checkpoint_dir):
    """Load mT5 model with LoRA adapter (no custom layers)"""
    print(f"\n🔍 Loading checkpoint: {checkpoint_dir}")

    # Load tokenizer from checkpoint
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    print(f"   Tokenizer vocab size: {len(tokenizer):,}")

    # Load base mT5 model
    print(f"   Loading base model: google/mt5-base")
    model = MT5ForConditionalGeneration.from_pretrained(
        "google/mt5-base",
        torch_dtype=torch.bfloat16
    )

    # Resize embeddings to match tokenizer
    print(f"   Original model vocab: {model.config.vocab_size:,}")
    model.resize_token_embeddings(len(tokenizer))
    print(f"   Resized model vocab:  {len(tokenizer):,}")

    # Load LoRA adapter from checkpoint
    print(f"   Loading LoRA adapter from: {checkpoint_dir}")
    model = PeftModel.from_pretrained(model, checkpoint_dir)
    
    print(f"✅ Loaded mT5 + LoRA successfully")
    
    return tokenizer, model




# =====================================================
# 2. MODEL ANALYSIS HELPERS
# =====================================================
def count_parameters(model):
    """Đếm số lượng parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    
    return {
        'total_params': total,
        'trainable_params': trainable,
        'frozen_params': frozen,
        'total_params_M': total / 1e6,
        'trainable_params_M': trainable / 1e6,
        'trainable_ratio': trainable / total if total > 0 else 0,
    }


def get_model_size_mb(model):
    """Tính kích thước model (MB)"""
    param_size = 0
    buffer_size = 0
    
    for p in model.parameters():
        param_size += p.nelement() * p.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    return (param_size + buffer_size) / (1024**2)


def measure_inference(model, tokenizer, texts, device, runs=100):
    """Đo thời gian inference và tính FPS"""
    times = []
    model.eval()
    
    print(f"\n⏱️  Đang đo inference time với {runs} samples...")

    with torch.no_grad():
        # Warmup
        for i in range(min(10, len(texts))):
            inp = tokenizer(texts[i], return_tensors="pt").to(device)
            _ = model.generate(**inp, max_length=64, num_beams=4)

        # Actual measurement
        for i in range(min(runs, len(texts))):
            inp = tokenizer(texts[i], return_tensors="pt").to(device)
            t0 = time.time()
            _ = model.generate(**inp, max_length=64, num_beams=4, early_stopping=True)
            t1 = time.time()
            times.append(t1 - t0)

    avg_time = np.mean(times)
    std_time = np.std(times)
    
    return {
        'avg_inference_time': avg_time,
        'std_inference_time': std_time,
        'min_inference_time': np.min(times),
        'max_inference_time': np.max(times),
        'avg_ms': avg_time * 1000,
        'min_ms': np.min(times) * 1000,
        'max_ms': np.max(times) * 1000,
        'fps': 1.0 / avg_time if avg_time > 0 else 0,
    }


def get_model_architecture_info(model, tokenizer):
    """Lấy thông tin kiến trúc model"""
    info = OrderedDict()
    
    info['model_type'] = 'MT5ForConditionalGeneration + LoRA'
    info['base_model'] = 'google/mt5-base'
    info['has_lora'] = hasattr(model, 'peft_config') or 'lora' in str(type(model)).lower()
    
    if hasattr(model, 'config'):
        config = model.config
        info['hidden_size'] = config.d_model if hasattr(config, 'd_model') else 'N/A'
        info['num_layers'] = config.num_layers if hasattr(config, 'num_layers') else 'N/A'
        info['num_heads'] = config.num_heads if hasattr(config, 'num_heads') else 'N/A'
        info['vocab_size'] = len(tokenizer)
        info['d_ff'] = config.d_ff if hasattr(config, 'd_ff') else 'N/A'
        info['num_decoder_layers'] = config.num_decoder_layers if hasattr(config, 'num_decoder_layers') else 'N/A'
    
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




# =====================================================
# 3. REPORT PRINTING
# =====================================================
def print_model_report(model, tokenizer, samples, device):
    """In báo cáo tổng hợp về model"""
    print("\n" + "="*80)
    print("📊 BÁO CÁO CHI TIẾT MÔ HÌNH MT5")
    print("="*80)
    
    # 1. Thông số parameters
    print("\n1️⃣  THÔNG SỐ PARAMETERS:")
    print("-" * 80)
    param_info = count_parameters(model)
    print(f"   • Tổng parameters:      {param_info['total_params']:,} ({param_info['total_params_M']:.2f}M)")
    print(f"   • Trainable parameters: {param_info['trainable_params']:,} ({param_info['trainable_params_M']:.2f}M)")
    print(f"   • Frozen parameters:    {param_info['frozen_params']:,}")
    print(f"   • Tỷ lệ trainable:      {param_info['trainable_ratio']*100:.2f}%")
    
    # 2. Kích thước model
    print("\n2️⃣  KÍCH THƯỚC MÔ HÌNH:")
    print("-" * 80)
    model_size = get_model_size_mb(model)
    print(f"   • Model size:           {model_size:.2f} MB")
    print(f"   • Tokenizer vocab size: {len(tokenizer):,}")
    
    # 3. Kiến trúc
    print("\n3️⃣  KIẾN TRÚC MÔ HÌNH:")
    print("-" * 80)
    arch_info = get_model_architecture_info(model, tokenizer)
    for key, value in arch_info.items():
        print(f"   • {key:.<30} {value}")
    
    # 4. Inference performance
    print("\n4️⃣  HIỆU SUẤT INFERENCE:")
    print("-" * 80)
    perf_info = measure_inference(model, tokenizer, samples, device)
    print(f"   • Avg inference time:   {perf_info['avg_ms']:.2f} ms")
    print(f"   • Std inference time:   {perf_info['std_inference_time']*1000:.2f} ms")
    print(f"   • Min inference time:   {perf_info['min_ms']:.2f} ms")
    print(f"   • Max inference time:   {perf_info['max_ms']:.2f} ms")
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


def save_model_report_to_file(report_data, output_file="model_report_mt5.txt"):
    """Lưu báo cáo ra file text"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("📊 BÁO CÁO CHI TIẾT MÔ HÌNH MT5\n")
        f.write("="*80 + "\n\n")
        
        f.write("1️⃣  THÔNG SỐ PARAMETERS:\n")
        f.write("-" * 80 + "\n")
        param_info = report_data['parameters']
        f.write(f"Tổng parameters:      {param_info['total_params']:,} ({param_info['total_params_M']:.2f}M)\n")
        f.write(f"Trainable parameters: {param_info['trainable_params']:,} ({param_info['trainable_params_M']:.2f}M)\n")
        f.write(f"Frozen parameters:    {param_info['frozen_params']:,}\n")
        f.write(f"Tỷ lệ trainable:      {param_info['trainable_ratio']*100:.2f}%\n\n")
        
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
        f.write(f"Avg inference time: {perf['avg_ms']:.2f} ms\n")
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
# 4. EVALUATE + METRICS + REPORT
# =====================================================
def evaluate_mt5(checkpoint_dir, test_csv):
    """Evaluate model và tạo báo cáo chi tiết"""
    tokenizer, model = load_model_mt5(checkpoint_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    df = pd.read_csv(test_csv)
    preds, refs = [], []
    test_texts = []

    print("\n🚀 Generating predictions...")
    for _, row in df.iterrows():
        text_in = row["input_text"]
        text_ref = row["target_text"]

        test_texts.append(text_in)

        inp = tokenizer(text_in, return_tensors="pt").to(device)
        
        with torch.no_grad():
            out = model.generate(
                **inp,
                max_length=64,
                num_beams=4,
                early_stopping=True
            )
        pred = tokenizer.decode(out[0], skip_special_tokens=True)

        preds.append(pred)
        refs.append(text_ref)

    # Compute metrics
    bleu = corpus_bleu([[r.split()] for r in refs], [p.split() for p in preds])
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

    r1 = np.mean([scorer.score(r, p)["rouge1"].fmeasure for p, r in zip(preds, refs)])
    rl = np.mean([scorer.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs)])

    print("\n====== 📊 EVALUATION METRICS ======")
    print(f"BLEU      : {bleu:.4f}")
    print(f"ROUGE-1 F1: {r1:.4f}")
    print(f"ROUGE-L F1: {rl:.4f}")

    # Save CSV results
    pd.DataFrame({
        "input_text": df["input_text"],
        "target_text": refs,
        "pred_text": preds
    }).to_csv("eval_results_mt5.csv", index=False)
    print("💾 Saved eval_results_mt5.csv")

    # Save metrics chart
    plt.figure(figsize=(8, 5))
    metrics = ["BLEU", "ROUGE-1", "ROUGE-L"]
    values = [bleu, r1, rl]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = plt.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add values on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.ylim(0, 1)
    plt.ylabel("Score", fontsize=12, fontweight='bold')
    plt.title("MT5 Model - Evaluation Metrics", fontsize=14, fontweight='bold')
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig("metrics_mt5.png", dpi=300, bbox_inches='tight')
    print("📊 Saved metrics_mt5.png")

    # Print detailed model report
    report = print_model_report(model, tokenizer, test_texts, device)
    
    # Save report to file
    save_model_report_to_file(report, "model_report_mt5.txt")
    
    # Create additional analysis charts
    plt.figure(figsize=(10, 6))
    param_info = report['parameters']
    
    # Subplot 1: Parameters distribution
    plt.subplot(1, 2, 1)
    sizes = [param_info['trainable_params_M'], param_info['total_params_M'] - param_info['trainable_params_M']]
    labels = ['Trainable', 'Frozen']
    colors_pie = ['#4ECDC4', '#95E1D3']
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors_pie, startangle=90)
    plt.title('Parameters Distribution', fontweight='bold')
    
    # Subplot 2: Performance metrics
    plt.subplot(1, 2, 2)
    perf_info = report['performance']
    perf_metrics = ['Avg Time\n(ms)', 'Min Time\n(ms)', 'Max Time\n(ms)']
    perf_values = [
        perf_info['avg_ms'],
        perf_info['min_ms'],
        perf_info['max_ms']
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
    plt.savefig("model_analysis_mt5.png", dpi=300, bbox_inches='tight')
    print("📊 Saved model_analysis_mt5.png")

    return bleu, r1, rl, report




# =====================================================
# 5. RUN
# =====================================================
if __name__ == "__main__":
    CHECKPOINT = "../weight/checkpoint-300"
    TEST_CSV = "test.csv"

    print("="*80)
    print("🚀 ĐÁNH GIÁ MÔ HÌNH MT5 VÀ TẠO BÁO CÁO CHI TIẾT")
    print("="*80)
    
    evaluate_mt5(CHECKPOINT, TEST_CSV)
    
    print("\n" + "="*80)
    print("✅ HOÀN TẤT! Đã tạo các file:")
    print("   📄 eval_results_mt5.csv     - Kết quả dự đoán chi tiết")
    print("   📊 metrics_mt5.png          - Biểu đồ các metrics (BLEU, ROUGE)")
    print("   📊 model_analysis_mt5.png   - Phân tích parameters và performance")
    print("   📝 model_report_mt5.txt     - Báo cáo chi tiết về model")
    print("="*80)

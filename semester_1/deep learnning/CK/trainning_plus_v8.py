"""
Áp dụng: Multi-Head Attention + BiLSTM + FFN
Model: vinai/phobert-base (~135M params)
"""

from transformers.modeling_outputs import SequenceClassifierOutput
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    TrainerCallback,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset, load_from_disk
import os
import json
from datetime import datetime
import warnings
import gc
import math
import matplotlib.pyplot as plt
from transformers.modeling_outputs import Seq2SeqLMOutput
warnings.filterwarnings('ignore')

# GLOBAL OPTIMIZATIONS
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# LOGGING
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

os.makedirs("./trained_phobert_dl/logs", exist_ok=True)
log_path = f"./trained_phobert_dl/logs/train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_file = open(log_path, "w", encoding="utf-8")

import sys
sys.stdout = Tee(sys.__stdout__, log_file)
sys.stderr = Tee(sys.__stderr__, log_file)

print("="*80)
print(f"📝 Logging to: {log_path}")
print("="*80)

# EFFICIENT MULTI-HEAD ATTENTION
class EfficientMultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert self.head_dim * num_heads == hidden_size
        
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.out = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_length, _ = hidden_states.size()
        
        qkv = self.qkv(hidden_states)
        qkv = qkv.view(batch_size, seq_length, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        
        Q, K, V = qkv[0], qkv[1], qkv[2]
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(attention_mask == 0, -1e4)
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_size)
        output = self.out(context)
        
        return output

# EFFICIENT BILSTM
class EfficientBiLSTM(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            num_layers=1,
            dropout=0,
            bidirectional=True,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, hidden_states):
        lstm_out, _ = self.lstm(hidden_states)
        output = self.layer_norm(hidden_states + self.dropout(lstm_out))
        return output

# EFFICIENT FFN
class EfficientFFN(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        ff_size = hidden_size * 2
        
        self.fc1 = nn.Linear(hidden_size, ff_size, bias=False)
        self.fc2 = nn.Linear(ff_size, hidden_size, bias=False)
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

# PHOBERT WITH CUSTOM LAYERS FOR SEQ2SEQ
class PhoBERTForSeq2Seq(nn.Module):
    """
    PhoBERT encoder + Custom layers + Decoder cho text generation
    """
    def __init__(self, base_model, vocab_size, hidden_size=None, num_heads=None):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        
        hidden_size = hidden_size or self.config.hidden_size
        num_heads = num_heads or self.config.num_attention_heads
        
        # Custom encoder layers
        self.multi_head_attention = EfficientMultiHeadAttention(
            hidden_size, num_heads, dropout=0.1
        )
        self.bilstm = EfficientBiLSTM(hidden_size, dropout=0.1)
        self.ffn = EfficientFFN(hidden_size, dropout=0.1)
        # Layer norms
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        print("✅ PhoBERT Seq2Seq Architecture:")
        print("   - PhoBERT Encoder (135M params)")
        print("   - Multi-Head Attention")
        print("   - Single-layer BiLSTM")
        print("   - Efficient FFN")
        print("   - LM Head for generation")
        
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        kwargs.pop("num_items_in_batch", None)
        
        # Encode với PhoBERT
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        hidden_states = outputs.last_hidden_state
        # Apply custom layers
        attn_output = self.multi_head_attention(hidden_states, attention_mask)
        attn_output = self.layer_norm1(hidden_states + self.dropout(attn_output))
        lstm_output = self.bilstm(attn_output)
        # Simple average
        ensemble_output = (attn_output + lstm_output) / 2
        ensemble_output = self.ffn(ensemble_output)
        final_hidden = self.layer_norm2(hidden_states + ensemble_output)
        # Generate logits
        logits = self.lm_head(final_hidden)
        # Compute loss
        loss = None
        if labels is not None:
            # Shift for autoregressive
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        # return Seq2SeqLMOutput(
        #     loss=loss,
        #     logits=logits,
        #     hidden_states=outputs.hidden_states
        # )
        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            encoder_last_hidden_state=hidden_states
        )

    def generate(self, input_ids, attention_mask=None, max_length=128, **kwargs):
        """Simple greedy generation"""
        batch_size = input_ids.size(0)
        device = input_ids.device
        generated = input_ids.clone()
        
        for _ in range(max_length - input_ids.size(1)):
            outputs = self.forward(
                input_ids=generated,
                attention_mask=attention_mask
            )
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            # Update attention mask
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((batch_size, 1), device=device)
                ], dim=1)
            # Stop if all sequences generated EOS
            if (next_token == 2).all():  # 2 = </s> token
                break
        return generated

# CUSTOM TRAINER
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss 
        if not hasattr(self, "train_losses"):
            self.train_losses = []
        # self.train_losses.append(loss.item())
        loss_scalar = loss.mean().detach().cpu().item()
        self.train_losses.append(loss_scalar)
        return (loss, outputs) if return_outputs else loss

# CALLBACKS
class SaveCustomLayersCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None:
            return control
        output_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        os.makedirs(output_dir, exist_ok=True)
        model_to_save = model.module if hasattr(model, "module") else model
        try:
            torch.save({
                "multi_head_attention": model_to_save.multi_head_attention.state_dict(),
                "bilstm": model_to_save.bilstm.state_dict(),
                "ffn": model_to_save.ffn.state_dict(),
                "lm_head": model_to_save.lm_head.state_dict(),
            }, os.path.join(output_dir, "custom_layers.pt"))
            print(f"💾 Saved custom_layers.pt at step {state.global_step}")
        except Exception as e:
            print(f"⚠️ Save error: {e}")
        return control

class MemoryMonitorCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 100 == 0:
            print(f"\n📊 Memory at Step {state.global_step}:")
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                print(f"   GPU {i}: {allocated:.2f}GB alloc, {reserved:.2f}GB reserved")
        return control

class AutoMemoryCleanupCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()
        return control

class TrainingVisualizationCallback(TrainerCallback):
    def __init__(self, trainer_ref, output_dir="./trained_phobert_dl/plots"):
        self.trainer_ref = trainer_ref
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def on_step_end(self, args, state, control, **kwargs):
        # ===== LƯU LOSS MỖI STEP ĐÚNG CHUẨN =====
        if hasattr(self.trainer_ref, "train_losses") and hasattr(self.trainer_ref.trainer, "train_losses"):
            # lấy loss từ CustomTrainer (chính xác)
            latest_losses = self.trainer_ref.trainer.train_losses
            if len(latest_losses) > 0:
                self.trainer_ref.train_losses.append(latest_losses[-1])
        return control

    def on_save(self, args, state, control, **kwargs):
        # ===== an toàn: tránh lỗi log_history trống =====
        if len(state.log_history) > 0:
            log_item = state.log_history[-1]
        else:
            log_item = {}
        # ===== GHI LOG THỜI GIAN AN TOÀN =====
        runtime = log_item.get("train_runtime", "n/a")
        with open(f"{self.output_dir}/time_log.txt", "a") as f:
            f.write(f"Step {state.global_step}: runtime={runtime}\n")
        # ===== VẼ BIỂU ĐỒ =====
        losses = self.trainer_ref.train_losses
        if len(losses) == 0:
            print("⚠️ No loss available to plot yet.")
            return control

        plt.figure(figsize=(10, 5))
        plt.plot(losses, label="Train Loss")
        plt.title(f"Train Loss up to Step {state.global_step}")
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.legend()
        out_path = f"{self.output_dir}/loss_step_{state.global_step}.png"
        plt.savefig(out_path)
        plt.close()
        print(f"📈 Saved loss plot: {out_path}")
        return control

# TRAINER CLASS
class PhoBERTTrainer:
    def __init__(self, model_name="vinai/phobert-base", output_dir="./trained_phobert_dl"):
        self.model_name = model_name
        self.output_dir = output_dir
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.train_losses = []
        self.start_time = None
        self.end_time = None
        self.inference_times = []
        os.makedirs(output_dir, exist_ok=True)
        self.setup_gpu()
    
    def setup_gpu(self):
        if not torch.cuda.is_available():
            print("❌ Không tìm thấy GPU!")
            exit(1)
        self.device = "cuda"
        self.n_gpu = torch.cuda.device_count()
        print(f"\n{'='*70}")
        print(f"🖥️  GPU INFORMATION")
        print(f"{'='*70}")
        for i in range(self.n_gpu):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   GPU {i}: {gpu_name}")
            print(f"   VRAM: {gpu_memory:.1f} GB")
        print(f"{'='*70}\n")
        torch.cuda.empty_cache()
        gc.collect()
        print("✅ Memory optimization enabled")
    
    def load_processed_data(self, data_dir="./processed_data"):
        print("="*70)
        print("BƯỚC 1: LOAD DỮ LIỆU")
        print("="*70)
        try:
            train_df = pd.read_csv(f"{data_dir}/train.csv")
            val_df = pd.read_csv(f"{data_dir}/val.csv")
            test_df = pd.read_csv(f"{data_dir}/test.csv")
        except FileNotFoundError as e:
            print(f"❌ Không tìm thấy file: {e}")
            exit(1)
        
        train_df = train_df.dropna(subset=['input_text', 'target_text'])
        val_df = val_df.dropna(subset=['input_text', 'target_text'])
        train_df = train_df[(train_df['input_text'].str.strip() != '') & 
                           (train_df['target_text'].str.strip() != '')]
        val_df = val_df[(val_df['input_text'].str.strip() != '') & 
                       (val_df['target_text'].str.strip() != '')]
        print(f"✅ Train: {len(train_df):,} samples")
        print(f"✅ Val:   {len(val_df):,} samples")
        print(f"✅ Test:  {len(test_df):,} samples")
        return train_df, val_df, test_df
    
    def setup_tokenizer(self):
        print("\n" + "="*70)
        print("BƯỚC 2A: SETUP TOKENIZER")
        print("="*70)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=True
        )
        print(f"✅ Tokenizer loaded. Vocab: {len(self.tokenizer)}")
        print(f"   PAD: {self.tokenizer.pad_token}")
        print(f"   EOS: {self.tokenizer.eos_token}")
        return self.tokenizer
    
    def setup_model(self):
        print("\n" + "="*70)
        print("BƯỚC 2B: SETUP PHOBERT MODEL")
        print("="*70)
        
        base_model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16
        )
        
        hidden_size = base_model.config.hidden_size
        num_heads = base_model.config.num_attention_heads
        vocab_size = len(self.tokenizer)
        
        print(f"\n🔧 Creating PhoBERT Seq2Seq...")
        print(f"   Hidden size: {hidden_size}")
        print(f"   Num heads: {num_heads}")
        print(f"   Vocab size: {vocab_size}")
        
        model = PhoBERTForSeq2Seq(
            base_model=base_model,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_heads=num_heads
        )
        
        # Multi-GPU
        if self.n_gpu > 1:
            print(f"\n🚀 Using DataParallel with {self.n_gpu} GPUs")
            model = nn.DataParallel(model)
            model = model.cuda()
        else:
            model = model.to(self.device)
        
        self.model = model
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print("\n" + "="*70)
        print("📊 MODEL INFO")
        print("="*70)
        print(f"   Total params: {total_params:,}")
        print(f"   Trainable params: {trainable_params:,}")
        print(f"   Model size (BF16): {total_params * 2 / 1024**3:.2f} GB")
        return self.model
    
    def tokenize_data(self, train_df, val_df, max_length=256):
        print("\n" + "="*70)
        print("BƯỚC 3: TOKENIZE DỮ LIỆU")
        print("="*70)
        tokenized_dir = f"{self.output_dir}/processed_tokenized"
        
        if os.path.exists(f"{tokenized_dir}/train") and os.path.exists(f"{tokenized_dir}/val"):
            print("📂 Loading cached...")
            tokenized_train = load_from_disk(f"{tokenized_dir}/train")
            tokenized_val = load_from_disk(f"{tokenized_dir}/val")
            print(f"✅ Train: {len(tokenized_train):,}")
            print(f"✅ Val: {len(tokenized_val):,}")
            return tokenized_train, tokenized_val
        
        def tokenize_function(examples):
            texts = [
                f"{inp} {tgt}"
                for inp, tgt in zip(examples['input_text'], examples['target_text'])
            ]
            tokenized = self.tokenizer(
                texts,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors=None
            )
            labels = []
            for ids in tokenized["input_ids"]:
                ids = ids.copy()
                labels.append([
                    (tok if tok != self.tokenizer.pad_token_id else -100)
                    for tok in ids
                ])
            tokenized["labels"] = labels
            return tokenized
        
        train_dataset = Dataset.from_pandas(train_df[['input_text', 'target_text']])
        val_dataset = Dataset.from_pandas(val_df[['input_text', 'target_text']])
        
        tokenized_train = train_dataset.map(
            tokenize_function,
            batched=True,
            batch_size=1000,
            num_proc=4,
            remove_columns=train_dataset.column_names,
            desc="Tokenizing train"
        )
        
        tokenized_val = val_dataset.map(
            tokenize_function,
            batched=True,
            batch_size=1000,
            num_proc=4,
            remove_columns=val_dataset.column_names,
            desc="Tokenizing val"
        )
        
        print(f"✅ Train: {len(tokenized_train):,}")
        print(f"✅ Val: {len(tokenized_val):,}")
        
        os.makedirs(tokenized_dir, exist_ok=True)
        tokenized_train.save_to_disk(f"{tokenized_dir}/train")
        tokenized_val.save_to_disk(f"{tokenized_dir}/val")
        
        return tokenized_train, tokenized_val
    
    def train(self, tokenized_train, tokenized_val, 
              num_epochs=3, batch_size=16, learning_rate=2e-5):
        print("\n" + "="*70)
        print("BƯỚC 4: TRAINING")
        print("="*70)
        # data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            max_length=256
        )        
        grad_accum = 2  # 16 * 2 GPUs * 2 = 64 effective batch
        steps_per_epoch = len(tokenized_train) // (batch_size * self.n_gpu * grad_accum)
        total_steps = steps_per_epoch * num_epochs
        warmup_steps = int(0.1 * total_steps)
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            logging_dir=f"{self.output_dir}/logs",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            gradient_accumulation_steps=grad_accum,
            learning_rate=learning_rate,
            weight_decay=0.01,
            max_grad_norm=1.0,
            warmup_steps=warmup_steps,
            num_train_epochs=num_epochs,
            fp16=False,
            bf16=True,
            do_eval=True,
            eval_strategy="steps",
            eval_steps=100,
            save_steps=100,
            save_total_limit=3,
            logging_steps=50,
            logging_first_step=True,
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            seed=42,
            disable_tqdm=False,
            remove_unused_columns=False,
            save_safetensors=False,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            report_to="none",
            optim="adamw_torch_fused",
        )
        
        print("\n📋 CONFIG:")
        print(f"   🎯 Model: {self.model_name}")
        print(f"   🎯 Batch per GPU: {batch_size}")
        print(f"   🎯 GPUs: {self.n_gpu}")
        print(f"   🎯 Effective batch: {batch_size * self.n_gpu * grad_accum}")
        print(f"   🎯 Total steps: {total_steps:,}")
        
        print("\n🚀 STARTING TRAINING...")
        print("="*70)
        save_callback = SaveCustomLayersCallback()
        memory_callback = MemoryMonitorCallback()
        cleanup_callback = AutoMemoryCleanupCallback()
        vis_callback = TrainingVisualizationCallback(self)
        self.trainer = CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            callbacks=[save_callback, memory_callback, cleanup_callback, vis_callback],
        )
        try:
            train_result = self.trainer.train()
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return None
        torch.cuda.empty_cache()
        gc.collect()
        
        print("\n✅ TRAINING COMPLETE!")
        print(f"⏱️  Time: {train_result.metrics['train_runtime']/60:.1f} min")
        print(f"📊 Final Loss: {train_result.metrics['train_loss']:.4f}")
        
        # Save model
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        final_dir = f"{self.output_dir}/final_model"
        model_to_save.base_model.save_pretrained(final_dir)
        
        torch.save({
            'multi_head_attention': model_to_save.multi_head_attention.state_dict(),
            'bilstm': model_to_save.bilstm.state_dict(),
            'ffn': model_to_save.ffn.state_dict(),
            'lm_head': model_to_save.lm_head.state_dict(),
        }, f"{final_dir}/custom_layers.pt")
        self.tokenizer.save_pretrained(final_dir)
        print(f"✅ Model saved: {final_dir}")
        return train_result

def main():
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║     PHOBERT OPTIMIZED TRAINING (Nhỏ gọn, tiết kiệm VRAM)           ║
    ║  PhoBERT + Attention + BiLSTM + FFN                                 ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    trainer = PhoBERTTrainer(
        model_name="vinai/phobert-base",
        output_dir="./trained_phobert_dl"
    )
    train_df, val_df, test_df = trainer.load_processed_data()
    tokenizer = trainer.setup_tokenizer()
    tokenized_train, tokenized_val = trainer.tokenize_data(
        train_df, val_df,
        max_length=256
    )
    model = trainer.setup_model()
    train_result = trainer.train(
        tokenized_train,
        tokenized_val,
        num_epochs=3,
        batch_size=16, 
        learning_rate=2e-5
    )
    if train_result:
        print("\n🎉 SUCCESS!")
        print("\n📝 Model Info:")
        print("   ✅ PhoBERT base (~135M params)")
        print("   ✅ Custom layers: ~10M params")
        print("   ✅ Total: ~145M params")
        print("   ✅ VRAM: ~6-8GB per GPU")
        print("   ✅ Batch size: 16")
if __name__ == "__main__":
    main()

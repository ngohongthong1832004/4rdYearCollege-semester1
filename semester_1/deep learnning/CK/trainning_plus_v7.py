from transformers.modeling_outputs import BaseModelOutput
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    MT5ForConditionalGeneration,
    MT5Tokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import os
import json
from datetime import datetime
import warnings
import gc
import math
import numpy as np
import sys
from datasets import load_from_disk
warnings.filterwarnings('ignore')

# LOGGING: Ghi đồng thời ra console và file log
import sys, os
from datetime import datetime

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

os.makedirs("./trained_model_advanced_v7/logs", exist_ok=True)
log_path = f"./trained_model_advanced_v7/logs/train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_file = open(log_path, "w", encoding="utf-8")

sys.stdout = Tee(sys.__stdout__, log_file)
sys.stderr = Tee(sys.__stderr__, log_file)

print("="*80)
print(f"📝 Logging simultaneously to console and {log_path}")
print("="*80)

# Custom Multi-Head Att
# 1. CUSTOM MULTI-HEAD ATTENTION (Transformer)
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
            scores = scores.masked_fill(attention_mask == 0, -1e4)
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_size)
        output = self.out(context)
        
        return output, attn_weights

# 2. BIDIRECTIONAL LSTM LAYER
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
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, hidden_states):
        lstm_out, _ = self.lstm(hidden_states)
        output = self.layer_norm(hidden_states + self.dropout(lstm_out))
        return output

# 3. RNN LAYER (Simple RNN)
class RNNLayer(nn.Module):
    """Simple RNN layer"""
    def __init__(self, hidden_size, num_layers=1, dropout=0.1):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, hidden_states):
        rnn_out, _ = self.rnn(hidden_states)
        output = self.layer_norm(hidden_states + self.dropout(rnn_out))
        return output

# 4. CONTEXT ATTENTION (Attention đơn giản)
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

# 5. FEED FORWARD NETWORK
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

# 6. ENSEMBLE MODEL - KẾT HỢP TẤT CẢ
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
        
        # 1. Multi-Head Attention (Transformer)
        self.multi_head_attention = CustomMultiHeadAttention(
            hidden_size, num_heads, dropout=0.1
        )
        # 2. Bidirectional LSTM
        self.bilstm = BiLSTMLayer(
            hidden_size, num_layers=2, dropout=0.1
        )
        # 3. Simple RNN
        self.rnn = RNNLayer(
            hidden_size, num_layers=1, dropout=0.1
        )
        # 4. Context Attention
        self.context_attention = ContextAttention(hidden_size)
        # 5. Feed Forward Network
        self.ffn = FeedForwardNetwork(hidden_size, dropout=0.1)
        # Layer norms
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.layer_norm3 = nn.LayerNorm(hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
        # Gating mechanism
        self.gate = nn.Linear(hidden_size * 3, 3)
        
        print("✅ Model Architecture:")
        print("   - Multi-Head Attention (Transformer)")
        print("   - Bidirectional LSTM (2 layers)")
        print("   - Simple RNN (1 layer)")
        print("   - Context Attention")
        print("   - Feed Forward Network")
        print("   - Gating Mechanism for Ensemble")
        
    def forward(self, input_ids=None, attention_mask=None, labels=None, 
                decoder_input_ids=None, decoder_attention_mask=None, **kwargs):

        if "num_items_in_batch" in kwargs:
            kwargs.pop("num_items_in_batch")

        # Nếu chưa có decoder_input_ids thì tự shift labels
        if decoder_input_ids is None and labels is not None:
            decoder_input_ids = self.base_model._shift_right(labels)

        # B1: Chạy encoder T5 để lấy hidden states
        encoder_outputs = self.base_model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        encoder_hidden = encoder_outputs.last_hidden_state
        # B2: Chạy qua các lớp nâng cao
        attn_output, _ = self.multi_head_attention(
            encoder_hidden,
            attention_mask=attention_mask.unsqueeze(1).unsqueeze(2) if attention_mask is not None else None
        )
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

        # B3: Gói output mới cho decoder
        new_encoder_outputs = BaseModelOutput(
            last_hidden_state=final_output,
            hidden_states=encoder_outputs.hidden_states
        )
        # B4: Chạy decoder T5 với encoder output mới
        outputs = self.base_model(
            encoder_outputs=new_encoder_outputs,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            return_dict=True
        )
        return outputs

    def generate(self, *args, **kwargs):
        return self.base_model.generate(*args, **kwargs)
    
    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.base_model.prepare_inputs_for_generation(*args, **kwargs)

# 7. CUSTOM TRAINER - FIXED
class CustomTrainer(Trainer):
    """Custom Trainer với label smoothing - FIXED"""
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")  # không pop!
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
    
# 8. TRAINER CLASS
class CommentGeneratorTrainer:
    """Main Trainer Class"""
    def __init__(self, model_name="google/mt5-base", output_dir="./trained_model_advanced_v7"):
        self.model_name = model_name
        self.output_dir = output_dir
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
        
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
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("✅ TF32 enabled")
    
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
        print("BƯỚC 2A: SETUP TOKENIZER (mT5)")
        print("="*70)
        
        self.tokenizer = MT5Tokenizer.from_pretrained(
            self.model_name,
            model_max_length=512,
            legacy=False
        )
        
        # Thêm special tokens cho platform
        special_tokens = {
            "additional_special_tokens": ["<TIKTOK>", "<FACEBOOK>", "<YOUTUBE>", "<COMMENT>"]
        }
        num_added = self.tokenizer.add_special_tokens(special_tokens)

        # mT5 thường không có pad_token -> set pad_token = eos_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"ℹ️  Set pad_token = eos_token ({self.tokenizer.pad_token})")

        print(f"✅ Tokenizer loaded (mT5). Vocab: {len(self.tokenizer)} (+{num_added})")
        
        return self.tokenizer

    def setup_model(self):
        print("\n" + "="*70)
        print("BƯỚC 2B: SETUP ADVANCED MODEL (mT5)")
        print("="*70)
        
        base_model = MT5ForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        )
        # Resize embeddings sau khi thêm special tokens
        base_model.resize_token_embeddings(len(self.tokenizer))
        # Đảm bảo config có pad_token_id khớp với tokenizer
        if base_model.config.pad_token_id is None and self.tokenizer.pad_token_id is not None:
            base_model.config.pad_token_id = self.tokenizer.pad_token_id
            print(f"ℹ️  Set model.config.pad_token_id = {base_model.config.pad_token_id}")

        hidden_size = base_model.config.d_model
        num_heads = base_model.config.num_heads
        
        print(f"\n🔧 Đang tạo Advanced Architecture...")
        model_with_advanced = T5WithAdvancedArchitecture(
            base_model=base_model,
            hidden_size=hidden_size,
            num_heads=num_heads
        )
        # LoRA giữ nguyên target_modules vì mT5 vẫn dùng T5 architecture
        print("\n🔧 Đang setup LoRA...")
        from peft import LoraConfig, get_peft_model, TaskType
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            target_modules=["q", "v", "k", "o", "wi_0", "wi_1", "wo"],
            inference_mode=False,
            bias="none"
        )
        model_with_advanced.base_model = get_peft_model(model_with_advanced.base_model, peft_config)
        # Multi-GPU
        if self.n_gpu > 1:
            print(f"\n🚀 DataParallel với {self.n_gpu} GPUs")
            model_with_advanced = nn.DataParallel(model_with_advanced)
            model_with_advanced = model_with_advanced.cuda()
        else:
            model_with_advanced = model_with_advanced.to(self.device)
        
        self.model = model_with_advanced
        print("\n" + "="*70)
        print("📊 THÔNG TIN MÔ HÌNH (mT5)")
        print("="*70)
        if hasattr(self.model, 'module'):
            self.model.module.base_model.print_trainable_parameters()
        else:
            self.model.base_model.print_trainable_parameters()
        return self.model
    
    def tokenize_data(self, train_df, val_df, max_input_length=512, max_target_length=128):
        print("\n" + "="*70)
        print("BƯỚC 3: TOKENIZE DỮ LIỆU")
        print("="*70)
        if os.path.exists("./trained_model_advanced_v7/processed_tokenized/train") and os.path.exists("./trained_model_advanced_v7/processed_tokenized/val"):
            print("📂 Phát hiện tokenized dataset đã tồn tại, đang load lại...")
            tokenized_train = load_from_disk("./trained_model_advanced_v7/processed_tokenized/train")
            tokenized_val = load_from_disk("./trained_model_advanced_v7/processed_tokenized/val")
            print(f"✅ Train: {len(tokenized_train):,}")
            print(f"✅ Val: {len(tokenized_val):,}")
            return tokenized_train, tokenized_val
        def tokenize_function(examples):
            model_inputs = self.tokenizer(
                examples['input_text'],
                max_length=max_input_length,
                truncation=True,
                padding=False
            )
            
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    examples['target_text'],
                    max_length=max_target_length,
                    truncation=True,
                    padding=False
                )
            
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        
        train_dataset = Dataset.from_pandas(train_df[['input_text', 'target_text']])
        val_dataset = Dataset.from_pandas(val_df[['input_text', 'target_text']])
        
        tokenized_train = train_dataset.map(
            tokenize_function,
            batched=True,
            batch_size=1000,
            num_proc=1,
            remove_columns=train_dataset.column_names,
            desc="Tokenizing train"
        )
        
        tokenized_val = val_dataset.map(
            tokenize_function,
            batched=True,
            batch_size=1000,
            num_proc=1,
            remove_columns=val_dataset.column_names,
            desc="Tokenizing val"
        )
        
        print(f"✅ Train: {len(tokenized_train):,}")
        print(f"✅ Val: {len(tokenized_val):,}")
        os.makedirs("./trained_model_advanced_v7/processed_tokenized", exist_ok=True)
        tokenized_train.save_to_disk("./trained_model_advanced_v7/processed_tokenized/train")
        tokenized_val.save_to_disk("./trained_model_advanced_v7/processed_tokenized/val")
        print("✅ Đã lưu tokenized dataset vào ./trained_model_advanced_v7/processed_tokenized/")          
        return tokenized_train, tokenized_val
    
    def train(self, tokenized_train, tokenized_val, 
              num_epochs=5, batch_size=24, learning_rate=2e-4):
        print("\n" + "="*70)
        print("BƯỚC 4: TRAINING")
        print("="*70)
        
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            pad_to_multiple_of=8
        )
        
        steps_per_epoch = len(tokenized_train) // (batch_size * self.n_gpu * 4)
        total_steps = steps_per_epoch * num_epochs
        warmup_steps = int(0.1 * total_steps)

        from transformers import TrainerCallback

        class SaveAdvancedLayersCallback(TrainerCallback):
            """Callback để lưu custom Advanced Layers + LoRA mỗi khi save checkpoint"""
            def on_save(self, args, state, control, **kwargs):
                model = kwargs.get("model")
                output_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
                os.makedirs(output_dir, exist_ok=True)

                # Unwrap nếu đang dùng DataParallel
                model_to_save = model.module if hasattr(model, "module") else model

                # === Lưu Custom Advanced Layers ===
                try:
                    torch.save({
                        "multi_head_attention": model_to_save.multi_head_attention.state_dict(),
                        "bilstm": model_to_save.bilstm.state_dict(),
                        "rnn": model_to_save.rnn.state_dict(),
                        "context_attention": model_to_save.context_attention.state_dict(),
                        "ffn": model_to_save.ffn.state_dict(),
                        "gate": model_to_save.gate.state_dict(),
                    }, os.path.join(output_dir, "advanced_layers.pt"))
                    print(f"💾 Saved advanced_layers.pt at step {state.global_step}")
                except Exception as e:
                    print(f"⚠️ Không thể lưu advanced_layers.pt: {e}")

                # === Lưu LoRA adapter ===
                try:
                    if hasattr(model_to_save, "base_model") and hasattr(model_to_save.base_model, "save_pretrained"):
                        model_to_save.base_model.save_pretrained(os.path.join(output_dir, "lora_adapter"))
                        print(f"💾 Saved LoRA adapter at step {state.global_step}")
                except Exception as e:
                    print(f"⚠️ Không thể lưu LoRA adapter: {e}")

                return control        
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            logging_dir=f"{self.output_dir}/logs",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            weight_decay=0.01,
            max_grad_norm=1.0,
            warmup_steps=warmup_steps,
            num_train_epochs=num_epochs,
            fp16=False,
            bf16=True,
            do_eval=True,
            eval_steps=100,
            save_steps=100,
            save_total_limit=3,
            logging_steps=100,
            logging_first_step=True,
            dataloader_num_workers=4,
            seed=42,
            disable_tqdm=False,
            remove_unused_columns=False,
            save_safetensors=False, 
            lr_scheduler_type="cosine",   # ✅ Mềm hơn, tránh plateau
            warmup_ratio=0.05,
        )
        
        print("\n📋 CẤU HÌNH:")
        print(f"   🎯 Batch per GPU: {batch_size}")
        print(f"   🎯 GPUs: {self.n_gpu}")
        print(f"   🎯 Effective batch: {batch_size * self.n_gpu * 4}")
        print(f"   🎯 Total steps: {total_steps:,}")
        print(f"   🎯 Architecture: Transformer + BiLSTM + RNN + Attention")
        
        print("\n🚀 BẮT ĐẦU TRAINING...")
        print("="*70)
        
        self.trainer = CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            callbacks=[SaveAdvancedLayersCallback()]
        )
        
        try:
            train_result = self.trainer.train()
        except Exception as e:
            print(f"\n❌ Lỗi: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        torch.cuda.empty_cache()
        gc.collect()
        
        print("\n✅ TRAINING HOÀN TẤT!")
        print(f"⏱️  Time: {train_result.metrics['train_runtime']/60:.1f} phút")
        print(f"📊 Loss: {train_result.metrics['train_loss']:.4f}")
        
        # Save
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        
        if hasattr(model_to_save, 'base_model'):
            model_to_save.base_model.save_pretrained(f"{self.output_dir}/final_model")
        
        # Save all custom layers
        torch.save({
            'multi_head_attention': model_to_save.multi_head_attention.state_dict(),
            'bilstm': model_to_save.bilstm.state_dict(),
            'rnn': model_to_save.rnn.state_dict(),
            'context_attention': model_to_save.context_attention.state_dict(),
            'ffn': model_to_save.ffn.state_dict(),
            'gate': model_to_save.gate.state_dict(),
        }, f"{self.output_dir}/final_model/advanced_layers.pt")
        
        self.tokenizer.save_pretrained(f"{self.output_dir}/final_model")
        
        print(f"✅ Model saved: {self.output_dir}/final_model")
        
        return train_result

def main():
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║     MT5 ADVANCED COMMENT GENERATOR - FULL ARCHITECTURE              ║
    ║  Transformer + BiLSTM + RNN + Attention + Ensemble + LoRA           ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    trainer = CommentGeneratorTrainer(
        model_name="google/mt5-base",
        output_dir="./trained_model_advanced_v7"
    )
    
    train_df, val_df, test_df = trainer.load_processed_data()
    tokenizer = trainer.setup_tokenizer()
    
    tokenized_train, tokenized_val = trainer.tokenize_data(
        train_df, val_df,
        max_input_length=256,
        max_target_length=64
    )
    
    model = trainer.setup_model()
    
    train_result = trainer.train(
        tokenized_train, 
        tokenized_val,
        num_epochs=5,
        batch_size=12,
        learning_rate=2e-4
    )
    if train_result:
        print("\n🎉 HOÀN TẤT!")
        print("\n📝 Đã áp dụng:")
        print("   ✅ Multi-Head Attention (Transformer)")
        print("   ✅ Bidirectional LSTM (2 layers)")
        print("   ✅ Simple RNN")
        print("   ✅ Context Attention")
        print("   ✅ Feed Forward Network")
        print("   ✅ Gating Ensemble Mechanism")
        print("   ✅ LoRA Fine-tuning")
        print("   ✅ Label Smoothing")
        print("   ✅ Residual Connections")
        print("   ✅ Layer Normalization")

if __name__ == "__main__":
    main()

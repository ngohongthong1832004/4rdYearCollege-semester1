import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers.modeling_outputs import BaseModelOutput
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers import (
    MT5ForConditionalGeneration,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import json
from datetime import datetime
import warnings
import gc
import math
import numpy as np
import sys

warnings.filterwarnings("ignore")


# ============================================================
# LOGGING
# ============================================================

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


os.makedirs("./trained_model_mt5/logs", exist_ok=True)
log_path = f"./trained_model_mt5/logs/train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_file = open(log_path, "w", encoding="utf-8")

sys.stdout = Tee(sys.__stdout__, log_file)
sys.stderr = Tee(sys.__stderr__, log_file)

print("=" * 80)
print(f"📝 Logging to: {log_path}")
print("=" * 80)


# ============================================================
# CALLBACKS
# ============================================================

import matplotlib.pyplot as plt

class TrainingVisualizationCallback(TrainerCallback):
    def __init__(self, output_dir="./trained_model_mt5/plots"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.logged_losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.logged_losses.append((state.global_step, float(logs["loss"])))
        return control

    def on_save(self, args, state, control, **kwargs):
        if len(self.logged_losses) < 5:
            return control

        steps, losses = zip(*self.logged_losses)

        plt.figure(figsize=(10, 5))
        plt.plot(steps, losses, label="Train Loss", linewidth=2)
        plt.grid(True)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.legend()

        out = f"{self.output_dir}/loss_step_{state.global_step}.png"
        plt.savefig(out)
        plt.close()

        print(f"📈 Saved loss plot → {out}")
        return control


class TimeLogCallback(TrainerCallback):
    def __init__(self, output_dir="./trained_model_mt5/plots"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def on_save(self, args, state, control, **kwargs):
        if state.log_history:
            line = f"Step {state.global_step} – time {state.log_history[-1].get('train_runtime','?')}\n"
            with open(f"{self.output_dir}/time_log.txt", "a") as f:
                f.write(line)
        return control

# ============================================================
# MAIN TRAINING CLASS
# ============================================================

class CommentGeneratorTrainer:
    def __init__(self, model_name="google/mt5-base", output_dir="./trained_model_mt5"):
        self.model_name = model_name
        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
        self.setup_gpu()

    def setup_gpu(self):
        if not torch.cuda.is_available():
            print("❌ NO GPU FOUND")
            exit()

        self.device = "cuda"
        self.n_gpu = torch.cuda.device_count()
        print("\n===== GPU INFO =====")
        for i in range(self.n_gpu):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print("====================\n")

    def load_processed_data(self, data_dir="./processed_data"):
        print("📥 Loading dataset...")

        train_df = pd.read_csv(f"{data_dir}/train.csv")
        val_df = pd.read_csv(f"{data_dir}/val.csv")
        test_df = pd.read_csv(f"{data_dir}/test.csv")

        train_df = train_df.dropna(subset=["input_text", "target_text"])
        val_df = val_df.dropna(subset=["input_text", "target_text"])

        print(f"Train: {len(train_df)}")
        print(f"Val:   {len(val_df)}")
        print(f"Test:  {len(test_df)}")

        return train_df, val_df, test_df

    def setup_tokenizer(self):
        print("Loading tokenizer...")

        saved_tok_dir = f"{self.output_dir}/tokenizer"
        if os.path.exists(saved_tok_dir):
            print(f"Loading tokenizer from {saved_tok_dir}")
            self.tokenizer = AutoTokenizer.from_pretrained(saved_tok_dir)
            print(f"Loaded vocab size: {len(self.tokenizer)}")
            return self.tokenizer

        print("Creating new tokenizer from base mT5")
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            model_max_length=512,
        )

        original_size = len(tokenizer)
        print(f"Original vocab: {original_size}")

        tokenizer.add_special_tokens({
            "additional_special_tokens": ["<TIKTOK>", "<FACEBOOK>", "<YOUTUBE>", "<COMMENT>"]
        })

        new_size = len(tokenizer)
        print(f"New vocab: {new_size} (added {new_size - original_size} tokens)")

        os.makedirs(saved_tok_dir, exist_ok=True)
        tokenizer.save_pretrained(saved_tok_dir)
        print(f"Saved tokenizer to {saved_tok_dir}")
        self.tokenizer = tokenizer
        return tokenizer


    def setup_model(self):
        print("🔧 Loading mT5 base model...")

        model = MT5ForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
        )

        print(f"Original embedding size: {model.get_input_embeddings().weight.shape}")
        
        # ✅ RESIZE VÀ KHỞI TẠO TOKEN MỚI ĐÚNG CÁCH
        model.resize_token_embeddings(len(self.tokenizer))
        
        # ✅ KHỞI TẠO EMBEDDING CHO TOKEN MỚI
        with torch.no_grad():
            input_embeddings = model.get_input_embeddings()
            output_embeddings = model.get_output_embeddings()
            
            # Lấy mean của embeddings cũ
            input_embeddings_avg = input_embeddings.weight[:-4].mean(dim=0)
            output_embeddings_avg = output_embeddings.weight[:-4].mean(dim=0)
            
            # Gán cho 4 token mới
            for i in range(4):
                input_embeddings.weight[-4 + i] = input_embeddings_avg
                output_embeddings.weight[-4 + i] = output_embeddings_avg
        
        print(f"New embedding size: {model.get_input_embeddings().weight.shape}")
        print(f"✅ Initialized {4} new token embeddings")

        # ✅ LORA CONFIG - CHỈ TRAIN LORA, KHÔNG TRAIN EMBEDDING
        peft_cfg = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q", "k", "v", "o", "wi_0", "wi_1", "wo"],
            inference_mode=False,
            modules_to_save=None,  # ✅ KHÔNG train embedding
        )

        model = get_peft_model(model, peft_cfg)
        model.print_trainable_parameters()

        model = model.to("cuda")
        
        print("✅ Model ready: mT5-base + LoRA")

        self.model = model
        return self.model

    def tokenize_data(self, train_df, val_df, max_input=256, max_target=64):
        print("✏️ Tokenizing dataset...")

        train_ds = Dataset.from_pandas(train_df[["input_text", "target_text"]])
        val_ds = Dataset.from_pandas(val_df[["input_text", "target_text"]])

        def encode(ex):
            inputs = self.tokenizer(
                ex["input_text"],
                truncation=True,
                max_length=max_input,
                padding=False,
            )

            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    ex["target_text"],
                    truncation=True,
                    max_length=max_target,
                    padding=False,
                )
            
            inputs["labels"] = labels["input_ids"]
            return inputs

        train_tok = train_ds.map(
            encode,
            batched=True,
            remove_columns=["input_text", "target_text"],
        )

        val_tok = val_ds.map(
            encode,
            batched=True,
            remove_columns=["input_text", "target_text"],
        )

        return train_tok, val_tok

    def train(self, train_tok, val_tok, epochs=4, batch=4, lr=3e-4):
        collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            pad_to_multiple_of=8,
        )

        # ✅ TRAINING ARGS - STABLE
        args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=batch,
            per_device_eval_batch_size=batch * 2,
            gradient_accumulation_steps=8,  # ✅ Tăng để effective batch = 32
            
            max_grad_norm=0.5,  # ✅ Giảm từ 1.0 → 0.5
            
            # Learning rate
            learning_rate=lr,
            lr_scheduler_type="cosine",
            warmup_steps=1000,  # ✅ Tăng warmup
            
            # Training
            num_train_epochs=epochs,
            logging_steps=10,
            
            # ✅ SAVE & EVAL STRATEGY
            save_strategy="steps",
            save_steps=300,
            save_total_limit=3,
            
            eval_strategy="steps",
            eval_steps=300,
            
            # Precision
            bf16=True,
            bf16_full_eval=True,
            
            # Evaluation
            do_eval=True,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            
            # Other - ✅ STABLE CONFIG
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
            
            # ✅ THÊM ĐỂ ỔN ĐỊNH
            optim="adamw_torch",
            weight_decay=0.01,
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_tok,
            eval_dataset=val_tok,
            tokenizer=self.tokenizer,
            data_collator=collator,
            callbacks=[
                TrainingVisualizationCallback(),
                TimeLogCallback(),
            ],
        )

        print("🚀 Training starts!")
        print(f"   Total steps: {len(train_tok) // (batch * 8) * epochs}")
        print(f"   Warmup steps: 1000")
        print(f"   Learning rate: {lr}")
        print(f"   Gradient clipping: 0.5")
        print(f"   Effective batch size: {batch * 8}")
        # print(f"   Eval & Save every: 500 steps")
        result = trainer.train()
        trainer.save_model(f"{self.output_dir}/final_model")
        self.tokenizer.save_pretrained(f"{self.output_dir}/final_model")
        print("🎉 Training finished!")
        print(f"💾 Model saved to: {self.output_dir}/final_model")
        return result

# MAIN
def main():
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    trainer = CommentGeneratorTrainer(
        model_name="google/mt5-base",
        output_dir="./trained_model_mt5",
    )

    train_df, val_df, test_df = trainer.load_processed_data()
    trainer.setup_tokenizer()
    train_tok, val_tok = trainer.tokenize_data(train_df, val_df)
    trainer.setup_model()
    
    trainer.train(
        train_tok,
        val_tok,
        epochs=4,
        batch=16,
        lr=3e-4,
    )
if __name__ == "__main__":
    main()

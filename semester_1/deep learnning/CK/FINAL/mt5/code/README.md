# Phân tích Chi tiết Code mT5 + LoRA

## Sơ đồ Tổng quan Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  INPUT → TOKENIZER → EMBEDDING → ENCODER → DECODER → LM HEAD → OUTPUT      │
│    ↓         ↓           ↓          ↓          ↓         ↓         ↓       │
│  String   [B,256]    [B,256,768] [B,256,768] [B,64,768] [B,64,V]  Loss     │
└─────────────────────────────────────────────────────────────────────────────┘

Pipeline đơn giản hơn - KHÔNG có Advanced Layers
Chỉ có: mT5 base + LoRA adapters
```

---

## 1. INPUT (Raw Text)

**Vị trí trong code:** `load_processed_data()`

```python
def load_processed_data(self, data_dir="./processed_data"):
    print("📥 Loading dataset...")

    train_df = pd.read_csv(f"{data_dir}/train.csv")
    val_df = pd.read_csv(f"{data_dir}/val.csv")
    test_df = pd.read_csv(f"{data_dir}/test.csv")

    # Loại bỏ rows có NaN
    train_df = train_df.dropna(subset=["input_text", "target_text"])
    val_df = val_df.dropna(subset=["input_text", "target_text"])

    return train_df, val_df, test_df
```

| Thuộc tính | Giá trị |
|------------|---------|
| **Kiểu dữ liệu** | `String` |
| **Columns** | `['input_text', 'target_text']` |
| **Ví dụ input** | `"Video này hay quá <TIKTOK>"` |
| **Ví dụ target** | `"Cảm ơn bạn đã chia sẻ!"` |

---

## 2. TOKENIZER

**Vị trí trong code:** `setup_tokenizer()` và `tokenize_data()`

```python
# ===== SETUP TOKENIZER =====
def setup_tokenizer(self):
    print("Loading tokenizer...")

    # Check nếu đã có tokenizer saved
    saved_tok_dir = f"{self.output_dir}/tokenizer"
    if os.path.exists(saved_tok_dir):
        print(f"Loading tokenizer from {saved_tok_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(saved_tok_dir)
        return self.tokenizer

    # Tạo mới từ mT5
    print("Creating new tokenizer from base mT5")
    tokenizer = AutoTokenizer.from_pretrained(
        self.model_name,        # "google/mt5-base"
        model_max_length=512,
    )

    original_size = len(tokenizer)  # ~250,000
    print(f"Original vocab: {original_size}")

    # Thêm 4 special tokens
    tokenizer.add_special_tokens({
        "additional_special_tokens": ["<TIKTOK>", "<FACEBOOK>", "<YOUTUBE>", "<COMMENT>"]
    })

    new_size = len(tokenizer)  # ~250,004
    print(f"New vocab: {new_size} (added {new_size - original_size} tokens)")

    # Save tokenizer
    tokenizer.save_pretrained(saved_tok_dir)
    self.tokenizer = tokenizer
    return tokenizer
```

```python
# ===== TOKENIZE DATA =====
def tokenize_data(self, train_df, val_df, max_input=256, max_target=64):
    print("✏️ Tokenizing dataset...")

    train_ds = Dataset.from_pandas(train_df[["input_text", "target_text"]])
    val_ds = Dataset.from_pandas(val_df[["input_text", "target_text"]])

    def encode(ex):
        # Tokenize input
        inputs = self.tokenizer(
            ex["input_text"],
            truncation=True,
            max_length=max_input,   # 256
            padding=False,
        )

        # Tokenize target
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                ex["target_text"],
                truncation=True,
                max_length=max_target,  # 64
                padding=False,
            )
        
        inputs["labels"] = labels["input_ids"]
        return inputs

    train_tok = train_ds.map(encode, batched=True, remove_columns=["input_text", "target_text"])
    val_tok = val_ds.map(encode, batched=True, remove_columns=["input_text", "target_text"])

    return train_tok, val_tok
```

| Input/Output | Shape | Giải thích |
|--------------|-------|------------|
| **Input** | `String` | Raw text |
| **Output (input_ids)** | `[batch_size, seq_len]` = `[16, 256]` | Token IDs |
| **Output (labels)** | `[batch_size, target_len]` = `[16, 64]` | Target token IDs |
| **Vocab size** | `250,004` | mT5 vocab + 4 special tokens |

**Ví dụ:**
```
"Video này hay <TIKTOK>" → [1234, 567, 89, 250001, 1]  (token IDs)
                                            ↑
                                     special token ID
```

---

## 3. EMBEDDING

**Vị trí trong code:** Nằm **bên trong** `MT5ForConditionalGeneration`, được xử lý trong `setup_model()`

```python
def setup_model(self):
    print("🔧 Loading mT5 base model...")

    model = MT5ForConditionalGeneration.from_pretrained(
        self.model_name,           # "google/mt5-base"
        torch_dtype=torch.bfloat16,
    )

    print(f"Original embedding size: {model.get_input_embeddings().weight.shape}")
    # Output: [250112, 768] (mT5 original vocab)
    
    # ✅ RESIZE EMBEDDING cho special tokens mới
    model.resize_token_embeddings(len(self.tokenizer))
    # New shape: [250004, 768]
    
    # ✅ KHỞI TẠO EMBEDDING CHO 4 TOKEN MỚI
    with torch.no_grad():
        input_embeddings = model.get_input_embeddings()
        output_embeddings = model.get_output_embeddings()
        
        # Lấy mean của tất cả embeddings cũ
        input_embeddings_avg = input_embeddings.weight[:-4].mean(dim=0)
        # Shape: [768] - vector trung bình
        
        output_embeddings_avg = output_embeddings.weight[:-4].mean(dim=0)
        # Shape: [768]
        
        # Gán mean cho 4 token mới
        for i in range(4):
            input_embeddings.weight[-4 + i] = input_embeddings_avg
            output_embeddings.weight[-4 + i] = output_embeddings_avg
    
    print(f"New embedding size: {model.get_input_embeddings().weight.shape}")
    # Output: [250004, 768]
```

**Embedding Layer (ẩn bên trong mT5):**
```python
# Pseudo-code của mT5 internal:
class MT5Stack:
    def __init__(self):
        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        # Shape: [250004, 768]
    
    def forward(self, input_ids):
        # input_ids: [batch, seq_len] = [16, 256]
        hidden_states = self.embed_tokens(input_ids)
        # hidden_states: [batch, seq_len, d_model] = [16, 256, 768]
        return hidden_states
```

| Thuộc tính | Shape | Giải thích |
|------------|-------|------------|
| **Input** | `[16, 256]` | Token IDs |
| **Embedding Matrix** | `[250004, 768]` | Vocab × Hidden |
| **Output** | `[16, 256, 768]` | Embedded vectors |
| **4 new tokens** | Initialized với mean | `<TIKTOK>`, `<FACEBOOK>`, `<YOUTUBE>`, `<COMMENT>` |

---

## 4. mT5 ENCODER (+ LoRA)

**Vị trí trong code:** Bên trong `MT5ForConditionalGeneration`, LoRA được apply trong `setup_model()`

```python
def setup_model(self):
    # ... sau phần embedding ...

    # ✅ LORA CONFIG
    peft_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=16,                    # LoRA rank
        lora_alpha=32,           # Scaling factor
        lora_dropout=0.1,
        target_modules=["q", "k", "v", "o", "wi_0", "wi_1", "wo"],
        inference_mode=False,
        modules_to_save=None,    # KHÔNG train embedding
    )

    # Apply LoRA vào model
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()
    # Output: trainable params: ~1.5M (0.6% of 250M)

    model = model.to("cuda")
    self.model = model
    return self.model
```

**Cấu trúc mT5 Encoder (12 layers):**

```
Input [16, 256, 768]
        ↓
┌─────────────────────────────────────────────────────────────┐
│  Layer 1-12 (mỗi layer có):                                 │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Self-Attention                                      │   │
│  │  ├─ Q = W_q × input + LoRA_A_q × LoRA_B_q × input   │   │
│  │  │   W_q: [768, 768], LoRA_A: [768, 16], LoRA_B: [16, 768]
│  │  ├─ K = W_k × input + LoRA_A_k × LoRA_B_k × input   │   │
│  │  ├─ V = W_v × input + LoRA_A_v × LoRA_B_v × input   │   │
│  │  └─ O = W_o × attn_out + LoRA_A_o × LoRA_B_o × attn_out
│  └─────────────────────────────────────────────────────┘   │
│                         ↓                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  LayerNorm                                           │   │
│  └─────────────────────────────────────────────────────┘   │
│                         ↓                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Feed-Forward Network (FFN)                          │   │
│  │  ├─ wi_0: [768, 2048] + LoRA  (gate)                │   │
│  │  ├─ wi_1: [768, 2048] + LoRA  (up projection)       │   │
│  │  └─ wo:   [2048, 768] + LoRA  (down projection)     │   │
│  │                                                      │   │
│  │  Computation: wo(gelu(wi_0(x)) * wi_1(x))           │   │
│  └─────────────────────────────────────────────────────┘   │
│                         ↓                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  LayerNorm                                           │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
        ↓
Output [16, 256, 768]
```

**LoRA Math:**
```
Original:  Y = W × X           where W: [768, 768]
With LoRA: Y = W × X + (A × B) × X
           where A: [768, 16], B: [16, 768]
           
Trainable params per module = 768 × 16 + 16 × 768 = 24,576
```

| Thuộc tính | Shape/Value |
|------------|-------------|
| **Input** | `[16, 256, 768]` |
| **Output** | `[16, 256, 768]` |
| **Num layers** | 12 |
| **Hidden size** | 768 |
| **FFN size** | 2048 |
| **Attention heads** | 12 |
| **LoRA rank (r)** | 16 |
| **LoRA alpha** | 32 |
| **Target modules** | q, k, v, o, wi_0, wi_1, wo |

---

## 5. mT5 DECODER (+ LoRA)

**Vị trí trong code:** Tự động được gọi khi forward model với labels

```python
# Khi training, Trainer tự động gọi:
outputs = model(
    input_ids=input_ids,           # [16, 256]
    attention_mask=attention_mask,  # [16, 256]
    labels=labels,                  # [16, 64]
)

# Bên trong model:
# 1. Encoder xử lý input_ids → encoder_hidden_states [16, 256, 768]
# 2. Decoder nhận labels (shifted) → decoder_hidden_states [16, 64, 768]
# 3. LM Head → logits [16, 64, 250004]
# 4. CrossEntropyLoss(logits, labels) → loss
```

**Cấu trúc mT5 Decoder (12 layers):**

```
labels [16, 64]
        ↓
    Shift Right → decoder_input_ids [16, 64]
    (thêm BOS token ở đầu)
        ↓
    Decoder Embedding [16, 64, 768]
        ↓
┌─────────────────────────────────────────────────────────────┐
│  Layer 1-12 (mỗi layer có):                                 │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Masked Self-Attention + LoRA                        │   │
│  │  (chỉ attend to previous tokens)                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                         ↓                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Cross-Attention + LoRA                              │   │
│  │  Q: from decoder [16, 64, 768]                       │   │
│  │  K, V: from encoder [16, 256, 768]                   │   │
│  └─────────────────────────────────────────────────────┘   │
│                         ↓                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  FFN + LoRA                                          │   │
│  │  [768 → 2048 → 768]                                  │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
        ↓
decoder_hidden_states [16, 64, 768]
```

| Thuộc tính | Shape/Value |
|------------|-------------|
| **Input (labels)** | `[16, 64]` |
| **decoder_input_ids** | `[16, 64]` (shifted right) |
| **Decoder embedding** | `[16, 64, 768]` |
| **Cross-attention to** | `[16, 256, 768]` (encoder output) |
| **Output** | `[16, 64, 768]` |

---

## 6. LM HEAD (Language Model Head)

**Vị trí trong code:** Bên trong `MT5ForConditionalGeneration`

```python
# Pseudo-code của mT5 internal:
class MT5ForConditionalGeneration:
    def __init__(self):
        # LM Head thường là tied với embedding (share weights)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Shape: [768, 250004]
        # Hoặc tied: self.lm_head.weight = self.shared.weight
    
    def forward(self, ...):
        # decoder_hidden: [16, 64, 768]
        
        # Project to vocab size
        lm_logits = self.lm_head(decoder_hidden)
        # lm_logits: [16, 64, 250004]
        
        return lm_logits
```

| Thuộc tính | Shape |
|------------|-------|
| **Input** | `[16, 64, 768]` |
| **LM Head weights** | `[768, 250004]` (tied với embedding) |
| **Output (logits)** | `[16, 64, 250004]` |

---

## 7. OUTPUT (Loss Computation)

**Vị trí trong code:** Tự động trong `Trainer` và bên trong model

```python
# Bên trong MT5ForConditionalGeneration.forward():
def forward(self, input_ids, attention_mask, labels, ...):
    # ... encoder, decoder ...
    
    lm_logits = self.lm_head(decoder_outputs)  # [16, 64, 250004]
    
    loss = None
    if labels is not None:
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Flatten
        loss = loss_fct(
            lm_logits.view(-1, self.config.vocab_size),  # [16*64, 250004] = [1024, 250004]
            labels.view(-1)                               # [1024]
        )
    
    return Seq2SeqLMOutput(
        loss=loss,
        logits=lm_logits,
        ...
    )
```

```python
# Training trong train():
def train(self, train_tok, val_tok, epochs=4, batch=16, lr=3e-4):
    # DataCollator xử lý padding
    collator = DataCollatorForSeq2Seq(
        tokenizer=self.tokenizer,
        model=self.model,
        pad_to_multiple_of=8,
    )

    # Training Arguments
    args = TrainingArguments(
        output_dir=self.output_dir,
        per_device_train_batch_size=batch,           # 16
        gradient_accumulation_steps=8,                # Effective batch = 128
        learning_rate=lr,                             # 3e-4
        lr_scheduler_type="cosine",
        warmup_steps=1000,
        num_train_epochs=epochs,                      # 4
        max_grad_norm=0.5,
        bf16=True,
        # ...
    )

    trainer = Trainer(
        model=self.model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=self.tokenizer,
        data_collator=collator,
        # ...
    )

    result = trainer.train()
    return result
```

| Thuộc tính | Value |
|------------|-------|
| **Loss function** | `CrossEntropyLoss(ignore_index=-100)` |
| **Logits shape** | `[16, 64, 250004]` → flatten `[1024, 250004]` |
| **Labels shape** | `[16, 64]` → flatten `[1024]` |
| **Output** | scalar loss |

---

## 8. FULL PIPELINE DIAGRAM

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          FULL mT5 + LoRA PIPELINE                            │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  "Video này hay <TIKTOK>"                                                   │
│           │                                                                  │
│           ▼                                                                  │
│  ┌─────────────────┐                                                        │
│  │    TOKENIZER    │  String → [16, 256] (token IDs)                        │
│  │   (AutoTokenizer)│  Vocab: 250,004 (mT5 + 4 special)                     │
│  └────────┬────────┘                                                        │
│           │                                                                  │
│           ▼                                                                  │
│  ┌─────────────────┐                                                        │
│  │    EMBEDDING    │  [16, 256] → [16, 256, 768]                            │
│  │   (inside mT5)  │  Matrix: [250004, 768]                                 │
│  │                 │  4 new tokens: initialized with mean                   │
│  └────────┬────────┘                                                        │
│           │                                                                  │
│           ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      mT5 ENCODER (12 layers)                         │    │
│  │                                                                      │    │
│  │   Input: [16, 256, 768]                                             │    │
│  │                                                                      │    │
│  │   Each layer:                                                        │    │
│  │   ┌────────────────────────────────────────────────────────────┐    │    │
│  │   │  Self-Attention                                             │    │    │
│  │   │  Q, K, V, O: [768, 768] + LoRA [768,16]×[16,768]           │    │    │
│  │   │  12 heads × 64 dim = 768                                    │    │    │
│  │   ├────────────────────────────────────────────────────────────┤    │    │
│  │   │  FFN                                                        │    │    │
│  │   │  wi_0, wi_1: [768, 2048] + LoRA                            │    │    │
│  │   │  wo: [2048, 768] + LoRA                                    │    │    │
│  │   │  Activation: GELU                                          │    │    │
│  │   └────────────────────────────────────────────────────────────┘    │    │
│  │                                                                      │    │
│  │   Output: [16, 256, 768]                                            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                │                                             │
│                                ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      mT5 DECODER (12 layers)                         │    │
│  │                                                                      │    │
│  │   labels [16, 64] → shift right → decoder_input_ids [16, 64]        │    │
│  │                         ↓                                            │    │
│  │   Decoder Embedding: [16, 64, 768]                                  │    │
│  │                         ↓                                            │    │
│  │   Each layer:                                                        │    │
│  │   ┌────────────────────────────────────────────────────────────┐    │    │
│  │   │  Masked Self-Attention + LoRA                               │    │    │
│  │   │  (causal: only attend to past tokens)                       │    │    │
│  │   ├────────────────────────────────────────────────────────────┤    │    │
│  │   │  Cross-Attention + LoRA                                     │    │    │
│  │   │  Q: from decoder [16, 64, 768]                              │    │    │
│  │   │  K, V: from encoder [16, 256, 768]                          │    │    │
│  │   ├────────────────────────────────────────────────────────────┤    │    │
│  │   │  FFN + LoRA                                                 │    │    │
│  │   │  [768 → 2048 → 768]                                         │    │    │
│  │   └────────────────────────────────────────────────────────────┘    │    │
│  │                                                                      │    │
│  │   Output: [16, 64, 768]                                             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                │                                             │
│                                ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                          LM HEAD                                     │    │
│  │                                                                      │    │
│  │   Linear: [768, 250004] (tied with embedding)                       │    │
│  │   Input:  [16, 64, 768]                                             │    │
│  │   Output: [16, 64, 250004] (logits)                                 │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                │                                             │
│                                ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                          OUTPUT                                      │    │
│  │                                                                      │    │
│  │   Training:                                                          │    │
│  │   └─ CrossEntropyLoss(logits.view(-1, 250004), labels.view(-1))     │    │
│  │   └─ [1024, 250004] vs [1024] → scalar loss                         │    │
│  │                                                                      │    │
│  │   Inference:                                                         │    │
│  │   └─ argmax(logits) → token IDs                                     │    │
│  │   └─ tokenizer.decode() → "Cảm ơn bạn đã chia sẻ!"                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Bảng Tổng hợp Tất cả Shapes

| # | Component | Code Location | Input Shape | Output Shape | Weights Shape |
|---|-----------|---------------|-------------|--------------|---------------|
| 1 | **Input** | `load_processed_data()` | String | String | - |
| 2 | **Tokenizer** | `setup_tokenizer()`, `tokenize_data()` | String | `[16, 256]` | Vocab: 250,004 |
| 3 | **Embedding** | Inside mT5 (`setup_model()`) | `[16, 256]` | `[16, 256, 768]` | `[250004, 768]` |
| 4 | **Encoder** | Inside mT5 | `[16, 256, 768]` | `[16, 256, 768]` | ~125M + LoRA |
| 5 | **Decoder** | Inside mT5 | `[16, 64]` + encoder | `[16, 64, 768]` | ~125M + LoRA |
| 6 | **LM Head** | Inside mT5 | `[16, 64, 768]` | `[16, 64, 250004]` | tied w/ embed |
| 7 | **Loss** | Inside mT5 | `[1024, 250004]` + `[1024]` | scalar | - |

---

## 10. LoRA Parameters Summary

**Target modules với LoRA:**

| Module | Original Shape | LoRA A | LoRA B | Trainable per module |
|--------|---------------|--------|--------|---------------------|
| `q` | `[768, 768]` | `[768, 16]` | `[16, 768]` | 24,576 |
| `k` | `[768, 768]` | `[768, 16]` | `[16, 768]` | 24,576 |
| `v` | `[768, 768]` | `[768, 16]` | `[16, 768]` | 24,576 |
| `o` | `[768, 768]` | `[768, 16]` | `[16, 768]` | 24,576 |
| `wi_0` | `[768, 2048]` | `[768, 16]` | `[16, 2048]` | 45,056 |
| `wi_1` | `[768, 2048]` | `[768, 16]` | `[16, 2048]` | 45,056 |
| `wo` | `[2048, 768]` | `[2048, 16]` | `[16, 768]` | 45,056 |

**Per layer:** ~233K params
**12 Encoder + 12 Decoder = 24 layers:** ~5.6M trainable params

```
Total LoRA trainable: ~1.5M (sau khi PEFT tối ưu)
Total model params: ~250M
Trainable ratio: ~0.6%
```

---

## 11. Training Config Summary

| Config | Value | Code Location |
|--------|-------|---------------|
| **Batch size** | 16 | `train(..., batch=16)` |
| **Gradient accumulation** | 8 | `gradient_accumulation_steps=8` |
| **Effective batch** | 128 | 16 × 8 |
| **Learning rate** | 3e-4 | `train(..., lr=3e-4)` |
| **Scheduler** | cosine | `lr_scheduler_type="cosine"` |
| **Warmup steps** | 1000 | `warmup_steps=1000` |
| **Epochs** | 4 | `train(..., epochs=4)` |
| **Max input length** | 256 | `tokenize_data(..., max_input=256)` |
| **Max target length** | 64 | `tokenize_data(..., max_target=64)` |
| **Precision** | bf16 | `bf16=True` |
| **Grad clipping** | 0.5 | `max_grad_norm=0.5` |
| **Weight decay** | 0.01 | `weight_decay=0.01` |
| **Optimizer** | AdamW | `optim="adamw_torch"` |

---

## 12. So sánh với Code Advanced (tóm tắt)

| Aspect | **mT5 + LoRA (Code này)** | **ViT5 + Advanced** |
|--------|--------------------------|---------------------|
| **Pipeline** | Simple: Enc → Dec | Complex: Enc → Advanced → Dec |
| **Advanced Layers** | ❌ Không có | ✅ Attn + BiLSTM + RNN + Gate + FFN |
| **Trainable params** | ~1.5M | ~18-20M |
| **Vocab size** | 250,004 | ~32,004 |
| **Ngôn ngữ tối ưu** | Multilingual | Tiếng Việt |
| **Training speed** | Nhanh hơn | Chậm hơn |
| **Memory usage** | Ít hơn | Nhiều hơn |
# Phân tích Chi tiết Từng Phần trong Pipeline

## Sơ đồ Tổng quan với Code Reference

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  INPUT → TOKENIZER → EMBEDDING → ENCODER → ADVANCED → DECODER → OUTPUT     │
│    ↓         ↓           ↓          ↓          ↓          ↓         ↓      │
│  String   [B,256]    [B,256,768] [B,256,768]  [B,256,768] [B,64,768] Logits │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. INPUT (Raw Text)

**Vị trí trong code:** `train_df`, `val_df`

```python
# Load data từ CSV
train_df = pd.read_csv(f"{data_dir}/train.csv")
# Columns: ['input_text', 'target_text']
```

| Thuộc tính | Giá trị |
|------------|---------|
| **Kiểu dữ liệu** | `String` |
| **Ví dụ input** | `"Video này hay quá <TIKTOK>"` |
| **Ví dụ target** | `"Cảm ơn bạn đã chia sẻ!"` |

---

## 2. TOKENIZER

**Vị trí trong code:** `setup_tokenizer()` và `tokenize_data()`

```python
# ===== SETUP TOKENIZER =====
def setup_tokenizer(self):
    self.tokenizer = T5Tokenizer.from_pretrained(
        self.model_name,  # "VietAI/vit5-base"
        model_max_length=512,
        legacy=False
    )
    
    # Thêm special tokens
    special_tokens = {
        "additional_special_tokens": ["<TIKTOK>", "<FACEBOOK>", "<YOUTUBE>", "<COMMENT>"]
    }
    num_added = self.tokenizer.add_special_tokens(special_tokens)
    return self.tokenizer

# ===== TOKENIZE DATA =====
def tokenize_function(examples):
    model_inputs = self.tokenizer(
        examples['input_text'],
        max_length=max_input_length,  # 256
        truncation=True,
        padding=False
    )
    
    with self.tokenizer.as_target_tokenizer():
        labels = self.tokenizer(
            examples['target_text'],
            max_length=max_target_length,  # 64
            truncation=True,
            padding=False
        )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
```

| Input/Output | Shape | Giải thích |
|--------------|-------|------------|
| **Input** | `String` | Raw text |
| **Output** | `[batch_size, seq_len]` = `[12, 256]` | Token IDs (integers) |
| **Labels** | `[batch_size, target_len]` = `[12, 64]` | Target token IDs |

**Ví dụ:**
```
"Video này hay" → [1234, 567, 89, 1]  (token IDs)
```

---

## 3. EMBEDDING

**Vị trí trong code:** Nằm **bên trong** `T5ForConditionalGeneration` (không viết explicit)

```python
# Embedding được tự động xử lý khi gọi encoder
# Nằm trong: base_model.encoder()

# Code resize embedding khi thêm special tokens:
base_model.resize_token_embeddings(len(self.tokenizer))
# Vocab: ~32,000 + 4 = ~32,004
```

**Embedding Layer (ẩn bên trong T5):**
```python
# Pseudo-code của T5 internal:
class T5Stack:
    def __init__(self):
        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        # Shape: [32004, 768]
    
    def forward(self, input_ids):
        # input_ids: [batch, seq_len] = [12, 256]
        hidden_states = self.embed_tokens(input_ids)
        # hidden_states: [batch, seq_len, d_model] = [12, 256, 768]
        return hidden_states
```

| Input/Output | Shape | Giải thích |
|--------------|-------|------------|
| **Input** | `[12, 256]` | Token IDs |
| **Embedding Matrix** | `[32004, 768]` | Vocab × Hidden |
| **Output** | `[12, 256, 768]` | Embedded vectors |

---

## 4. T5 ENCODER

**Vị trí trong code:** Trong `forward()` của `T5WithAdvancedArchitecture`

```python
def forward(self, input_ids=None, attention_mask=None, labels=None, ...):
    # ...
    
    # ===== T5 ENCODER =====
    encoder_outputs = self.base_model.encoder(
        input_ids=input_ids,           # [12, 256]
        attention_mask=attention_mask,  # [12, 256]
        output_hidden_states=True,
        return_dict=True
    )
    
    encoder_hidden = encoder_outputs.last_hidden_state  # [12, 256, 768]
```

**Cấu trúc T5 Encoder (12 layers):**
```
Input [12, 256, 768]
        ↓
┌─────────────────────────────────┐
│  Layer 1-12 (mỗi layer có):     │
│  ├─ Self-Attention + LoRA       │
│  │   Q, K, V, O: [768, 768]     │
│  ├─ LayerNorm                   │
│  ├─ FFN (wi_0, wi_1, wo) + LoRA │
│  │   [768→2048→768]             │
│  └─ LayerNorm                   │
└─────────────────────────────────┘
        ↓
Output [12, 256, 768]
```

| Input/Output | Shape |
|--------------|-------|
| **Input** | `[12, 256, 768]` |
| **Output** | `[12, 256, 768]` |

---

## 5. ADVANCED LAYERS

### 5.1 Multi-Head Attention

**Vị trí trong code:** Class `CustomMultiHeadAttention`

```python
class CustomMultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        # hidden_size = 768, num_heads = 12
        self.head_dim = hidden_size // num_heads  # 768 // 12 = 64
        
        self.query = nn.Linear(hidden_size, hidden_size)  # [768, 768]
        self.key = nn.Linear(hidden_size, hidden_size)    # [768, 768]
        self.value = nn.Linear(hidden_size, hidden_size)  # [768, 768]
        self.out = nn.Linear(hidden_size, hidden_size)    # [768, 768]
        
        self.scale = math.sqrt(self.head_dim)  # sqrt(64) = 8
        
    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_length, _ = hidden_states.size()  # 12, 256, 768
        
        # Linear projections
        Q = self.query(hidden_states)  # [12, 256, 768]
        K = self.key(hidden_states)    # [12, 256, 768]
        V = self.value(hidden_states)  # [12, 256, 768]
        
        # Reshape to multi-head: [batch, heads, seq, head_dim]
        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        # [12, 256, 12, 64] → [12, 12, 256, 64]
        K = K.view(...).transpose(1, 2)  # [12, 12, 256, 64]
        V = V.view(...).transpose(1, 2)  # [12, 12, 256, 64]
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # [12, 12, 256, 64] @ [12, 12, 64, 256] = [12, 12, 256, 256]
        
        attn_weights = torch.softmax(scores, dim=-1)  # [12, 12, 256, 256]
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)  # [12, 12, 256, 64]
        
        # Reshape back
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_size)
        # [12, 12, 256, 64] → [12, 256, 12, 64] → [12, 256, 768]
        
        output = self.out(context)  # [12, 256, 768]
        return output, attn_weights
```

| Bước | Shape | Giải thích |
|------|-------|------------|
| Input | `[12, 256, 768]` | encoder_hidden |
| Q, K, V | `[12, 256, 768]` | Linear projection |
| Reshape | `[12, 12, 256, 64]` | Split heads |
| Scores | `[12, 12, 256, 256]` | Q @ K^T |
| Context | `[12, 12, 256, 64]` | Scores @ V |
| **Output** | `[12, 256, 768]` | Merge heads |

---

### 5.2 BiLSTM

**Vị trí trong code:** Class `BiLSTMLayer`

```python
class BiLSTMLayer(nn.Module):
    def __init__(self, hidden_size, num_layers=2, dropout=0.1):
        # hidden_size = 768
        self.lstm = nn.LSTM(
            input_size=hidden_size,      # 768
            hidden_size=hidden_size // 2, # 384 (mỗi hướng)
            num_layers=num_layers,        # 2
            dropout=dropout,
            bidirectional=True,           # Forward + Backward
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_size)  # [768]
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, hidden_states):
        # hidden_states: [12, 256, 768]
        
        lstm_out, _ = self.lstm(hidden_states)
        # lstm_out: [12, 256, 768]  (384 forward + 384 backward = 768)
        
        # Residual connection + LayerNorm
        output = self.layer_norm(hidden_states + self.dropout(lstm_out))
        # output: [12, 256, 768]
        
        return output
```

| Bước | Shape | Giải thích |
|------|-------|------------|
| Input | `[12, 256, 768]` | encoder_hidden |
| LSTM Forward | `[12, 256, 384]` | Left-to-right |
| LSTM Backward | `[12, 256, 384]` | Right-to-left |
| Concat | `[12, 256, 768]` | 384 + 384 |
| **Output** | `[12, 256, 768]` | After residual |

---

### 5.3 RNN

**Vị trí trong code:** Class `RNNLayer`

```python
class RNNLayer(nn.Module):
    def __init__(self, hidden_size, num_layers=1, dropout=0.1):
        # hidden_size = 768
        self.rnn = nn.RNN(
            input_size=hidden_size,   # 768
            hidden_size=hidden_size,  # 768
            num_layers=num_layers,    # 1
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, hidden_states):
        # hidden_states: [12, 256, 768]
        
        rnn_out, _ = self.rnn(hidden_states)
        # rnn_out: [12, 256, 768]
        
        output = self.layer_norm(hidden_states + self.dropout(rnn_out))
        # output: [12, 256, 768]
        
        return output
```

| Input/Output | Shape |
|--------------|-------|
| Input | `[12, 256, 768]` |
| **Output** | `[12, 256, 768]` |

---

## 6. GATING MECHANISM

**Vị trí trong code:** Trong `forward()` của `T5WithAdvancedArchitecture`

```python
# Trong __init__:
self.gate = nn.Linear(hidden_size * 3, 3)  # [2304, 3]

# Trong forward():
def forward(self, ...):
    # ... sau khi có 3 outputs từ advanced layers
    
    # attn_output: [12, 256, 768]
    # lstm_output: [12, 256, 768]
    # rnn_output:  [12, 256, 768]
    
    # Concatenate
    combined = torch.cat([attn_output, lstm_output, rnn_output], dim=-1)
    # combined: [12, 256, 2304]  (768 * 3)
    
    # Compute gate weights
    gates = torch.softmax(self.gate(combined), dim=-1)
    # gates: [12, 256, 3]  (3 weights cho 3 branches)
    
    # Weighted ensemble
    ensemble_output = (
        gates[:, :, 0:1] * attn_output +   # [12, 256, 1] * [12, 256, 768]
        gates[:, :, 1:2] * lstm_output +   # [12, 256, 1] * [12, 256, 768]
        gates[:, :, 2:3] * rnn_output      # [12, 256, 1] * [12, 256, 768]
    )
    # ensemble_output: [12, 256, 768]
```

| Bước | Shape | Giải thích |
|------|-------|------------|
| 3 Inputs | `[12, 256, 768]` × 3 | Attn, LSTM, RNN |
| Concat | `[12, 256, 2304]` | 768 × 3 |
| Gate Linear | `[12, 256, 3]` | 3 weights |
| Softmax | `[12, 256, 3]` | Sum = 1 |
| **Output** | `[12, 256, 768]` | Weighted sum |

---

## 7. FEED FORWARD NETWORK (FFN)

**Vị trí trong code:** Class `FeedForwardNetwork`

```python
class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ff_size=None, dropout=0.1):
        # hidden_size = 768
        ff_size = ff_size or hidden_size * 4  # 3072
        
        self.fc1 = nn.Linear(hidden_size, ff_size)   # [768, 3072]
        self.fc2 = nn.Linear(ff_size, hidden_size)   # [3072, 768]
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)  # [768]
        
    def forward(self, hidden_states):
        # hidden_states: [12, 256, 768]
        
        residual = hidden_states
        
        x = F.gelu(self.fc1(hidden_states))  # [12, 256, 3072]
        x = self.dropout(x)
        x = self.fc2(x)                       # [12, 256, 768]
        x = self.dropout(x)
        
        output = self.layer_norm(residual + x)  # [12, 256, 768]
        return output
```

| Bước | Shape | Giải thích |
|------|-------|------------|
| Input | `[12, 256, 768]` | ensemble_output |
| fc1 + GELU | `[12, 256, 3072]` | Expand |
| fc2 | `[12, 256, 768]` | Contract |
| **Output** | `[12, 256, 768]` | After residual |

---

## 8. FINAL OUTPUT (trước Decoder)

**Vị trí trong code:** Cuối phần Advanced trong `forward()`

```python
def forward(self, ...):
    # ... sau FFN
    
    ensemble_output = self.ffn(ensemble_output)  # [12, 256, 768]
    
    # Final residual connection
    final_output = self.layer_norm3(encoder_hidden + ensemble_output)
    # final_output: [12, 256, 768]
    
    # Wrap thành BaseModelOutput cho decoder
    new_encoder_outputs = BaseModelOutput(
        last_hidden_state=final_output,          # [12, 256, 768]
        hidden_states=encoder_outputs.hidden_states
    )
```

---

## 9. T5 DECODER

**Vị trí trong code:** Cuối `forward()`

```python
def forward(self, ...):
    # Tạo decoder_input_ids nếu chưa có
    if decoder_input_ids is None and labels is not None:
        decoder_input_ids = self.base_model._shift_right(labels)
        # labels: [12, 64] → decoder_input_ids: [12, 64]
        # Shift right: [BOS, tok1, tok2, ...] thay vì [tok1, tok2, ..., EOS]
    
    # Chạy full model với encoder output mới
    outputs = self.base_model(
        encoder_outputs=new_encoder_outputs,     # [12, 256, 768]
        attention_mask=attention_mask,           # [12, 256]
        decoder_input_ids=decoder_input_ids,     # [12, 64]
        decoder_attention_mask=decoder_attention_mask,
        labels=labels,                           # [12, 64]
        return_dict=True
    )
    
    return outputs
    # outputs.logits: [12, 64, vocab_size] = [12, 64, 32004]
    # outputs.loss: scalar
```

**Cấu trúc T5 Decoder (12 layers):**
```
decoder_input_ids [12, 64]
        ↓
    Embedding [12, 64, 768]
        ↓
┌─────────────────────────────────────┐
│  Layer 1-12 (mỗi layer có):         │
│  ├─ Masked Self-Attention + LoRA    │
│  ├─ Cross-Attention + LoRA          │
│  │   (attend to encoder output)     │
│  └─ FFN + LoRA                      │
└─────────────────────────────────────┘
        ↓
    Hidden [12, 64, 768]
        ↓
    LM Head [12, 64, 32004]
```

| Bước | Shape | Giải thích |
|------|-------|------------|
| decoder_input_ids | `[12, 64]` | Shifted labels |
| Decoder embedding | `[12, 64, 768]` | Token → Vector |
| Cross-attention | Attends to `[12, 256, 768]` | encoder output |
| **Logits** | `[12, 64, 32004]` | Vocab probabilities |

---

## 10. OUTPUT (Loss & Generation)

**Training mode:**
```python
# Loss được tính tự động trong T5
# outputs.loss = CrossEntropyLoss(logits, labels)

loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
loss = loss_fct(
    logits.view(-1, vocab_size),  # [12*64, 32004] = [768, 32004]
    labels.view(-1)               # [768]
)
```

**Inference mode:**
```python
def generate(self, *args, **kwargs):
    return self.base_model.generate(*args, **kwargs)

# Sử dụng:
output_ids = model.generate(
    input_ids=input_ids,      # [1, 256]
    max_length=64,
    num_beams=4,
    ...
)
# output_ids: [1, generated_length]
# Decode: tokenizer.decode(output_ids[0])
```

---

## Tổng hợp: Flow Chart với Shapes

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FULL PIPELINE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  "Video này hay <TIKTOK>"                                                   │
│           │                                                                  │
│           ▼                                                                  │
│  ┌─────────────────┐                                                        │
│  │    TOKENIZER    │  String → [12, 256] (token IDs)                        │
│  └────────┬────────┘                                                        │
│           │                                                                  │
│           ▼                                                                  │
│  ┌─────────────────┐                                                        │
│  │    EMBEDDING    │  [12, 256] → [12, 256, 768]                            │
│  │  (inside T5)    │  Matrix: [32004, 768]                                  │
│  └────────┬────────┘                                                        │
│           │                                                                  │
│           ▼                                                                  │
│  ┌─────────────────┐                                                        │
│  │   T5 ENCODER    │  [12, 256, 768] → [12, 256, 768]                       │
│  │   (12 layers)   │  + LoRA on Q,K,V,O,FFN                                 │
│  └────────┬────────┘                                                        │
│           │                                                                  │
│           ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      ADVANCED LAYERS                                 │    │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐            │    │
│  │  │  MULTI-HEAD   │  │    BiLSTM     │  │     RNN       │            │    │
│  │  │   ATTENTION   │  │  (2 layers)   │  │  (1 layer)    │            │    │
│  │  │ [12,256,768]  │  │ [12,256,768]  │  │ [12,256,768]  │            │    │
│  │  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘            │    │
│  │          │                  │                  │                     │    │
│  │          └──────────────────┼──────────────────┘                     │    │
│  │                             ▼                                        │    │
│  │                    ┌─────────────────┐                               │    │
│  │                    │     GATING      │  [12,256,2304] → [12,256,768] │    │
│  │                    │  g0*A + g1*L    │  Gate: [2304, 3]              │    │
│  │                    │     + g2*R      │                               │    │
│  │                    └────────┬────────┘                               │    │
│  │                             │                                        │    │
│  │                             ▼                                        │    │
│  │                    ┌─────────────────┐                               │    │
│  │                    │      FFN        │  [12,256,768] → [12,256,768]  │    │
│  │                    │  768→3072→768   │  fc1: [768,3072]              │    │
│  │                    └────────┬────────┘  fc2: [3072,768]              │    │
│  │                             │                                        │    │
│  └─────────────────────────────┼────────────────────────────────────────┘    │
│                                │                                             │
│                                ▼                                             │
│                    final_output [12, 256, 768]                              │
│                                │                                             │
│                                ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        T5 DECODER                                    │    │
│  │  decoder_input_ids: [12, 64]                                        │    │
│  │           │                                                          │    │
│  │           ▼                                                          │    │
│  │  Decoder Embedding: [12, 64, 768]                                   │    │
│  │           │                                                          │    │
│  │           ▼                                                          │    │
│  │  12 Decoder Layers:                                                  │    │
│  │    - Masked Self-Attention                                          │    │
│  │    - Cross-Attention (to encoder [12,256,768])                      │    │
│  │    - FFN                                                             │    │
│  │           │                                                          │    │
│  │           ▼                                                          │    │
│  │  Hidden: [12, 64, 768]                                              │    │
│  │           │                                                          │    │
│  │           ▼                                                          │    │
│  │  LM Head: [768, 32004]                                              │    │
│  │           │                                                          │    │
│  │           ▼                                                          │    │
│  │  Logits: [12, 64, 32004]                                            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                │                                             │
│                                ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                          OUTPUT                                      │    │
│  │  Training: CrossEntropyLoss(logits, labels) → scalar loss           │    │
│  │  Inference: argmax(logits) → token IDs → decode → "Cảm ơn bạn!"     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Bảng Tổng hợp Tất cả Shapes

| # | Component | Code Location | Input Shape | Output Shape | Weights Shape |
|---|-----------|---------------|-------------|--------------|---------------|
| 1 | **Input** | `train_df` | String | String | - |
| 2 | **Tokenizer** | `tokenize_function()` | String | `[12, 256]` | Vocab: 32004 |
| 3 | **Embedding** | Inside T5 | `[12, 256]` | `[12, 256, 768]` | `[32004, 768]` |
| 4 | **T5 Encoder** | `base_model.encoder()` | `[12, 256, 768]` | `[12, 256, 768]` | ~110M |
| 5a | **Multi-Head Attn** | `CustomMultiHeadAttention` | `[12, 256, 768]` | `[12, 256, 768]` | 4×`[768,768]` |
| 5b | **BiLSTM** | `BiLSTMLayer` | `[12, 256, 768]` | `[12, 256, 768]` | ~7.1M |
| 5c | **RNN** | `RNNLayer` | `[12, 256, 768]` | `[12, 256, 768]` | ~1.2M |
| 6 | **Gating** | `self.gate` | `[12, 256, 2304]` | `[12, 256, 3]` | `[2304, 3]` |
| 7 | **FFN** | `FeedForwardNetwork` | `[12, 256, 768]` | `[12, 256, 768]` | `[768,3072]`+`[3072,768]` |
| 8 | **T5 Decoder** | `base_model()` | `[12, 64]` + encoder | `[12, 64, 768]` | ~110M |
| 9 | **LM Head** | Inside T5 | `[12, 64, 768]` | `[12, 64, 32004]` | `[768, 32004]` |
| 10 | **Loss** | `CrossEntropyLoss` | `[768, 32004]` + `[768]` | scalar | - |






Tôi yêu em

<start> ?  ?  ? ?           => Tôi, Mày, Tao, Nó => Beam => Tôi
<start> Tôi ? ? ?           => yêu => y
<start> Tôi yêu ? ?         => em
<start> Tôi Yêu em ?        => <End> =>  Dừng


[0 -inf -inf -inf]
[0 1 -inf -inf]
[0 1 2 -inf]
[0 1 2 3]



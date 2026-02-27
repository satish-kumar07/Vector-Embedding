## Project Overview

This project implements the **core input pipeline of a GPT-style Transformer model** from scratch using PyTorch.

It covers:

* Byte Pair Encoding (GPT-2 tokenizer)
* Sliding Window Data Sampling
* Custom PyTorch Dataset
* DataLoader batching
* Token Embedding
* Positional Embedding
* Final Input Embedding construction

The goal of this project is to deeply understand how raw text is converted into vector representations before entering a Transformer model.

---

# What This Project Implements

### 1️⃣ Tokenization 

We use OpenAI’s GPT-2 tokenizer via `tiktoken` to convert raw text into token IDs.

```python
tokenizer = tiktoken.get_encoding("gpt2")
token_ids = tokenizer.encode(txt)
```

---

### 2️⃣ Sliding Window Data Sampling

To train language models efficiently, the dataset is split into overlapping sequences using a sliding window approach.

Input sequence:

```
[t1, t2, t3, t4]
```

Target sequence:

```
[t2, t3, t4, t5]
```

This allows the model to learn **next-token prediction**.

---

### 3️⃣ Custom GPT Dataset

```python
class GPTDatasetV1(Dataset):
```

* Tokenizes full text
* Creates overlapping chunks
* Generates input-target shifted pairs
* Returns PyTorch tensors

---

### 4️⃣ DataLoader

```python
dataloader = DataLoader(dataset, batch_size=8)
```

Outputs batches of shape:

```
(batch_size, sequence_length)
```

Example:

```
torch.Size([8, 4])
```

---

### 5️⃣ Token Embedding

We create a learnable embedding matrix:

```python
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
```

Embedding matrix shape:

```
(50257, 256)
```

Meaning:

* 50,257 tokens
* Each mapped to a 256-dimensional vector

Passing token IDs through embedding:

```
Input shape:  (8, 4)
Output shape: (8, 4, 256)
```

---

### 6️⃣ Positional Embedding

Transformers do not understand sequence order naturally.

We add positional embeddings:

```python
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
```

Shape:

```
(4, 256)
```

---

### 7️⃣ Final Input Representation

The final embedding is computed as:

```
Input Embedding = Token Embedding + Positional Embedding
```

Final shape:

```
(8, 4, 256)
```

This matches the expected input format of Transformer models.

---

# 📊 Complete Data Flow

```
Raw Text
   ↓
GPT-2 Tokenizer
   ↓
Token IDs
   ↓
Sliding Window Sampling
   ↓
PyTorch Dataset
   ↓
DataLoader (Batching)
   ↓
Token Embedding
   ↓
Positional Embedding
   ↓
Final Input Embedding
```

---

# 🔍 Sample Output

Example token IDs batch:

```
tensor([
 [  40,  367, 2885, 1464],
 ...
])
```

Input shape:

```
torch.Size([8, 4])
```

Embedding output shape:

```
torch.Size([8, 4, 256])
```



# 📁 How to Run

1. Install dependencies:

```bash
pip install torch tiktoken
```

2. Place your text file (e.g., `the-verdict.txt`) in the project folder.

3. Run the notebook or script.

---

# 📜 License

This project is for educational purposes.

---

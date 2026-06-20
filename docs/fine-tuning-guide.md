# Fine-Tuning Guide — Tools, Techniques & Configuration

## Framework Comparison (February 2026)

| Framework | Speed | VRAM Efficiency | Ease of Use | Best For |
|---|---|---|---|---|
| **Unsloth** | 2-5x faster | 70% less VRAM | Easy | Consumer GPUs (recommended) |
| **LLaMA-Factory** | Standard | Good | Very easy (GUI) | Beginners, visual interface preference |
| **Axolotl** | Standard | Good | Medium complexity | Complex pipelines, distributed training |
| **torchtune** | 24% slower | Good | Medium | PyTorch purists, Meta ecosystem |
| **HF TRL + PEFT** | Standard | Standard | Easy | Ecosystem compatibility priority |

### Recommendation: **Unsloth**

Unsloth remains the gold standard for consumer GPU fine-tuning:
- **Speed:** 2-5x faster training (reduces 12-hour training to 3-4 hours)
- **VRAM:** 70% less VRAM overhead vs HF TRL
- **Maintenance:** Actively maintained, updated February 2026 with Llama 4, Gemma 3, vision model support
- **API:** Simple, clean interface — minimal boilerplate
- **Community:** Large, helpful community with many examples

**Alternative:** LLaMA-Factory if you prefer a GUI interface. Both are excellent, Unsloth is faster.

## PEFT (Parameter Efficient Fine-Tuning) Techniques

| Technique | VRAM for 7B | Quality vs Full FT | Speed | Notes |
|---|---|---|---|---|
| **QLoRA (4-bit)** | ~6-7GB | 80-90% | Good | Industry standard, recommended for 12GB GPU |
| **LoRA (16-bit)** | ~16GB | 90-95% | Good | Too much VRAM for 12GB (needs 16GB+ GPU) |
| **DoRA** | ~6-7GB | 92-95% | Good | Weight-decomposed, new (2025), worth trying |
| **GaLore** | Very low | 85-90% | Moderate | Gradient-focused efficiency, experimental |
| Full fine-tuning (16-bit) | ~48GB | 100% | Good | NOT viable on consumer hardware |

### Deep Dive: QLoRA (Quantized LoRA)

**Why it works:**
- Base model stored in 4-bit (nf4 quantization)
- LoRA adapters in 16-bit (small matrices, ~0.5GB)
- During forward pass: 4-bit model is dequantized on-the-fly to compute gradients
- Backward pass: gradients computed in 16-bit, stored in LoRA

**Result:** 11-12GB total VRAM for 7B model fine-tuning (barely fits in RTX 5070)

### Emerging: DoRA (Decomposed Rank-Adapted)

**New in 2025**, DoRA decomposes LoRA update into magnitude and direction:
- Same VRAM as LoRA/QLoRA
- Better quality convergence (92-95% vs 80-90%)
- Slightly more compute (negligible impact)
- Still experimental, but worth trying if Unsloth supports it

**Use if:** Quality is critical, hardware allows
**Use QLoRA if:** Pushing VRAM limits, want proven technique

## QLoRA Configuration for RTX 5070 (12GB)

### Minimal Example (Unsloth)

```python
from unsloth import FastLanguageModel
import torch

max_seq_length = 512
dtype = torch.bfloat16

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-8B-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)
```

### Full Training Configuration

```python
from transformers import TrainingArguments
from trl import SFTTrainer

training_args = TrainingArguments(
    per_device_train_batch_size=1,          # MUST be 1 on 12GB
    gradient_accumulation_steps=8,          # effective batch = 8
    warmup_steps=10,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=False,
    bf16=True,                              # RTX 5070 supports bfloat16
    optim="paged_adamw_8bit",               # 8-bit optimizer saves VRAM
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="outputs",
    report_to="none",                       # Skip wandb/huggingface tracking
    logging_steps=10,
    save_strategy="epoch",
    max_grad_norm=1.0,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    packing=True,                           # Pack multiple samples per sequence
    dataset_num_proc=2,
)

trainer.train()
```

### Memory Budget Breakdown (7B QLoRA on 12GB)

```
Base model (4-bit, nf4):    ~4.5 GB
LoRA adapters (16-bit):     ~0.5 GB
Optimizer states (8-bit):   ~1.5 GB
Gradients (16-bit):         ~1.0 GB
Activations (fp32):         ~3.0 GB (with gradient checkpointing)
CUDA overhead:              ~1.0 GB
─────────────────────────────────
Total:                      ~11.5 GB  ← fits in 12GB (tight!)
```

**Critical settings for 12GB:**
- `per_device_train_batch_size=1` (non-negotiable)
- `gradient_checkpointing=True` (saves activation memory)
- `bf16=True` (32-bit activations instead of FP32)
- `paged_adamw_8bit` (8-bit optimizer, not full precision)

### Sequence Length Trade-off

```
Max Sequence Length | VRAM | Training Time | Quality
────────────────────┼──────┼───────────────┼─────────
256                 | 8 GB | 2-3 hours     | Baseline
512                 | 10 GB| 4-6 hours     | Good
1024                | 12 GB| 8-12 hours    | Better
2048                | OOM  | N/A           | N/A (doesn't fit)
```

**Recommendation:** Start at 512, increase to 1024 if training completes successfully. Avoid 2048 on 12GB.

## Synthetic Dataset Creation

### Tool Comparison

| Tool | Type | Cost | Effort | Quality | Best For |
|---|---|---|---|---|---|
| **HF Synthetic Data Generator** | No-code, browser | Free | 10 min | Medium | Quick start, small datasets |
| **Meta Synthetic Data Kit** | Python pipeline | Free | 1 hour | High | Production Q&A, scalable |
| **NVIDIA Nemotron-4 340B** | Free API | Free | 30 min | Very high | Large-scale synthesis |
| **Claude API distillation** | API-based | $1-5 per 1K | 2 hours | Excellent | Highest quality (costs money) |
| **Local generation (Qwen 3)** | Local LLM | Free | 1-2 hours | Good | Self-contained, no API |

### Best Practices

1. **Mix synthetic + real data** — 60% synthetic, 40% real yields best results
2. **Minimum dataset size** — 1K examples for noticeable style change
3. **Target size** — 5-10K examples for meaningful domain adaptation
4. **Quality over quantity** — 1K high-quality examples > 10K low-quality
5. **Always include system prompts** — System context improves fine-tuning
6. **Diversify response lengths** — Mix short (50 tokens) and long (500+ tokens) responses

### Implementation: Local Generation with Qwen 3

```python
from ollama import Client

client = Client(host='http://localhost:11434')

def generate_qa_pair(topic: str) -> dict:
    """Generate a Q&A pair using Qwen 3 8B."""

    prompt = f"""Generate a question and answer about: {topic}

    Question should be specific and clear.
    Answer should be detailed (100-300 words).
    Format as JSON:
    {{"question": "...", "answer": "..."}}"""

    response = client.generate(
        model='qwen3:8b',
        prompt=prompt,
        stream=False,
    )

    import json
    return json.loads(response['response'])

# Generate dataset
dataset = []
topics = ["machine learning", "quantum computing", "climate science", ...]
for topic in topics:
    for _ in range(5):  # 5 Q&A pairs per topic
        qa = generate_qa_pair(topic)
        dataset.append(qa)

# Save
import json
with open('datasets/training/synthetic.jsonl', 'w') as f:
    for qa in dataset:
        f.write(json.dumps(qa) + '\n')
```

## Dataset Format: ChatML

Standard chat/instruction format used across Ollama, Hugging Face, and modern LLMs:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful AI assistant specializing in technical explanations."
    },
    {
      "role": "user",
      "content": "What is quantum entanglement?"
    },
    {
      "role": "assistant",
      "content": "Quantum entanglement is a phenomenon where two or more quantum particles become correlated..."
    }
  ]
}
```

### Conversion Script: JSON → ChatML

```python
def json_to_chatml(input_file: str, output_file: str):
    """Convert Q&A JSON to ChatML format."""
    import json

    with open(input_file) as f:
        data = json.load(f)

    with open(output_file, 'w') as out:
        for item in data:
            message = {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": item["question"]},
                    {"role": "assistant", "content": item["answer"]}
                ]
            }
            out.write(json.dumps(message) + '\n')

# Usage
json_to_chatml('datasets/raw/qa_pairs.json', 'datasets/training/chatml.jsonl')
```

### Train/Validation Split

```python
import json
from sklearn.model_selection import train_test_split

with open('datasets/training/chatml.jsonl') as f:
    data = [json.loads(line) for line in f]

train, val = train_test_split(data, test_size=0.1, random_state=42)

with open('datasets/training/train.jsonl', 'w') as f:
    for item in train:
        f.write(json.dumps(item) + '\n')

with open('datasets/training/val.jsonl', 'w') as f:
    for item in val:
        f.write(json.dumps(item) + '\n')

print(f"Train: {len(train)}, Val: {len(val)}")
```

## Inference Servers (2026 Options)

| Server | Throughput | Setup | Best For | Notes |
|---|---|---|---|---|
| **Ollama** | ~484 tok/s | Easiest (native Windows) | Local dev, model switching | Default for exploration |
| **llama.cpp** | ~200 tok/s | Easy, minimal deps | Portability, CPU inference | Best for MacOS/Linux |
| **vLLM** | ~8,033 tok/s | Medium (WSL2 or Docker) | Production API, batching | Distributed serving |
| **SGLang** | ~6,395 tok/s | Medium (WSL2 or Docker) | High throughput, structured output | RadixAttention optimization |
| **TensorRT-LLM** | ~1,406 tok/s | Hard (CUDA SDK required) | Maximum RTX 5070 performance | Overkill for local dev |

### Quick Comparison: Peak Performance

```
Model: Qwen 3 8B (Q4_K_M)
GPU: RTX 5070

Ollama:           50-70 tok/s      (easy, conservative)
llama.cpp:        40-60 tok/s      (portable)
vLLM (batch=8):   200-250 tok/s    (optimized)
SGLang (opt):     180-220 tok/s    (RadixAttention)
TensorRT-LLM:     350+ tok/s       (maximum, overkill)
```

### Recommendation Path

**Phase 1 (Exploration):** Ollama
- Easiest setup, native Windows
- `ollama serve` exposes API on port 11434
- Perfect for local development
- No performance tuning needed

**Phase 4 (Production):** SGLang or vLLM
- Requires WSL2 + Docker or Linux VM
- Better throughput for concurrent requests
- Structured output support (JSON schemas)
- Load balancing capabilities

### Ollama Setup (Phase 1)

```bash
# Download and install Ollama from ollama.ai
# Then in any terminal:
ollama serve

# In another terminal, test:
curl http://localhost:11434/api/generate \
  -d '{"model":"qwen3:8b","prompt":"Hello"}'

# Or use Python:
from ollama import Client
client = Client()
response = client.generate(model='qwen3:8b', prompt='Hello')
print(response['response'])
```

### Network Access (Phase 1)

Ollama listens on `127.0.0.1:11434` by default (localhost only).

To access from other machines:

**Option 1: Reverse Proxy (Recommended)**
```bash
# Use ngrok for temporary public access
ngrok http 11434

# Or use your router's port forwarding for home network
# Forward router port 11434 → machine's 127.0.0.1:11434
```

**Option 2: Bind to 0.0.0.0 (Less Secure)**
```bash
# In WSL2 or Linux:
OLLAMA_HOST=0.0.0.0:11434 ollama serve

# Note: This exposes your GPU to network attacks, use firewall
```

## Platform Setup: Inference vs Fine-tuning

### Inference (Phase 1) — Native Windows

**Ollama:**
1. Download installer from https://ollama.ai
2. Install (adds to PATH)
3. Open terminal: `ollama serve`
4. Verify: `ollama list`
5. Pull model: `ollama pull qwen3:8b`
6. Test: Python client or curl

**No WSL2 needed.** Ollama abstracts CUDA setup.

### Fine-tuning (Phase 3) — WSL2 Required

Windows 11 native CUDA development is possible but complex. WSL2 + Ubuntu simplifies setup:

**1. Enable WSL2**
```powershell
# In PowerShell (admin):
wsl --install
# Restart, open "Ubuntu 22.04" from Microsoft Store
```

**2. Install NVIDIA CUDA in WSL2**
```bash
# In WSL2 terminal:
sudo apt update
sudo apt install nvidia-cuda-toolkit

# Verify:
nvidia-smi  # Should show RTX 5070
```

**3. Install Unsloth**
```bash
pip install unsloth
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify:
python -c "from unsloth import FastLanguageModel; print('OK')"
```

**4. Test Training Script**
```bash
cd /path/to/code-ai-workspace
python fine-tuning/train.py
```

**Note:** VRAM is shared between Windows and WSL2. Close Ollama before training.

## Important Caveats

### Fine-tuning Teaches Style, Not Facts

Fine-tuning adjusts response format, tone, and writing style. It does NOT add factual knowledge from your documents.

**For factual domain adaptation:**
- Use **RAG (Retrieval Augmented Generation)** combined with fine-tuning
- Store documents in vector database (ChromaDB, Pinecone)
- At inference: retrieve relevant docs, include in prompt context
- Fine-tuning: train on desired response format for that domain

**Fine-tuning alone:** Good for instruction following, poor for new domain knowledge.

### 12GB is the Hard Minimum

- Current config fits in 12GB, but with zero slack
- `per_device_train_batch_size=1` is non-negotiable
- Sequence length > 1024 risks OOM
- Reduce batch size further if OOM: use gradient_accumulation_steps instead

### Training Time

On RTX 5070 with 5-10K examples:
- 1 epoch: 2-4 hours
- 3 epochs: 6-12 hours
- 5 epochs: 10-20 hours

Longer sequences = longer training.

### Sequence Length Guidelines

Start conservative, increase gradually:
```
Phase 1: 256 tokens — validate pipeline works (30 min training)
Phase 2: 512 tokens — real training (4-6 hours)
Phase 3: 1024 tokens — maximum quality (10-14 hours)
Phase 4: 2048 tokens — skip, OOM on 12GB
```

### If Out of Memory (OOM)

**In priority order:**
1. Reduce `max_seq_length` (256 → 128)
2. Enable `flash_attention_2` (if supported by model)
3. Reduce `r` (LoRA rank: 16 → 8)
4. Reduce `per_device_train_batch_size` (already 1, can't go lower)

**Do NOT:**
- Disable gradient_checkpointing (saves 30% VRAM)
- Switch to full precision (needs 4x VRAM)

## Saving and Deploying Fine-tuned Models

### After Training: Save LoRA Weights

```python
model.save_pretrained("fine-tuning/outputs/qwen3-8b-lora")
tokenizer.save_pretrained("fine-tuning/outputs/qwen3-8b-lora")
```

### Option 1: Keep LoRA Separate (Recommended)

**Advantage:** Modularity, can swap LoRAs without reloading base model
**Use case:** A/B testing different fine-tunes

```python
# At inference time with Ollama:
# (Ollama doesn't support LoRA yet — requires manual setup)

# With Unsloth:
model, _ = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-8B-bnb-4bit",
)
model = PeftModel.from_pretrained(model, "fine-tuning/outputs/qwen3-8b-lora")
model = model.merge_and_unload()  # Then quantize and serve
```

### Option 2: Merge LoRA into Base Model

**Advantage:** Single file, no LoRA loading overhead
**Use case:** Production deployment

```python
# Merge LoRA into base model
merged_model = model.merge_and_unload()
merged_model.save_pretrained("fine-tuning/outputs/qwen3-8b-merged")
tokenizer.save_pretrained("fine-tuning/outputs/qwen3-8b-merged")

# Convert to GGUF for Ollama (requires llama.cpp tools)
# Then: ollama create qwen3-8b-finetuned -f Modelfile
```

### Quantize Fine-tuned Model (Optional)

After merging, quantize to Q4_K_M for Ollama:

```bash
# Requires llama.cpp (GitHub: ggerganov/llama.cpp)
./quantize merged_model/model.gguf merged_model_q4.gguf Q4_K_M
```

This compresses merged model from ~16GB (FP32) to ~4.5GB (Q4).

## Training Monitoring & Validation

### Log Loss During Training

Unsloth + Hugging Face Trainer automatically logs to console. Watch for:
- **Validation loss decreasing:** Good convergence
- **Validation loss flat:** Model has plateaued, reduce learning_rate or num_epochs
- **Validation loss increasing:** Overfitting, increase dropout or reduce num_epochs

### Evaluate on Test Set

```python
from torch.utils.data import DataLoader
from tqdm import tqdm

def evaluate(model, test_dataset, tokenizer):
    model.eval()
    total_loss = 0
    for batch in DataLoader(test_dataset, batch_size=1):
        with torch.no_grad():
            outputs = model(**batch)
            total_loss += outputs.loss.item()
    return total_loss / len(test_dataset)

test_loss = evaluate(model, eval_dataset, tokenizer)
print(f"Test loss: {test_loss:.3f}")
```

Lower test loss = better fine-tuning. Compare to baseline (pre-trained model on same dataset).

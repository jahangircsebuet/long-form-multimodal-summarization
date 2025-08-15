# long-form-multimodal-summarization
Efficient Long-Form Multimodal Summarization (Hierarchy + Distill + Budgeted Decoding)


# Efficient Long-Form Multimodal Summarization
**Hierarchy + Distillation + Budgeted Decoding**

This repo implements a pragmatic pipeline for long/Noisy multimodal threads:
1) **Hierarchical summarization** (chunk → local summaries → dedupe → fused global).
2) **Retrieval-aware distillation (LoRA)** to compress a teacher into a student.
3) **Budgeted decoding** to hit strict token/latency budgets with minimal quality loss.

---

## 📁 Project Structure
## Project Structure

```text
├─ src/
│  ├─ segmentation/
│  │  ├─ init.py
│  │  └─ chunk.py
│  ├─ teacher/
│  │  ├─ init.py
│  │  ├─ local_summarizer.py
│  │  └─ fuse.py
│  ├─ distill/
│  │  ├─ init.py
│  │  ├─ make_dataset.py
│  │  └─ train_lora_seq2seq.py
│  ├─ decoding/
│  │  ├─ init.py
│  │  └─ budget.py
│  └─ utils/
│     ├─ init.py
│     └─ cost.py
├─ scripts/
│  ├─ run_hierarchy.py
│  └─ train_student_lora.py
└─ tests_integration_idea2/
   ├─ test_hierarchy_flow.py
   ├─ test_budget_decoder.py
   └─ test_tiny_distill_training.py



---

## ⚙️ Setup

```bash
# (Recommended) create a fresh GPU env
conda create -n idea2 python=3.10 -y
conda activate idea2

# PyTorch (+ CUDA if available)
# pick the right index for your CUDA version, e.g. cu121
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Core libs
pip install transformers accelerate datasets peft sentence-transformers faiss-cpu pynvml psutil pytest

# Optional: speed up tokenizers
pip install tokenizers

# Verify GPU is visible (optional)
python - << 'PY'
import torch; print("cuda?", torch.cuda.is_available(), "device_count:", torch.cuda.device_count())
PY


1) Prepare a long thread file

Put your long text into data/long_thread.txt (or any path).

python scripts/run_hierarchy.py \
  --input data/long_thread.txt \
  --out_dir outputs/hierarchy \
  --max_tokens 1024 --overlap 128 \
  --local_len 128 --global_len 200

Artifacts:

outputs/hierarchy/global_summary.txt — fused global summary.

outputs/hierarchy/local_summaries.txt — deduped local summaries (bullets).

outputs/hierarchy/distill_data/ — tiny distillation dataset (HF DatasetDict).



2) Train a LoRA student (tiny run)
python scripts/train_student_lora.py \
  --data_dir outputs/hierarchy/distill_data \
  --out_dir outputs/distill \
  --max_steps 200


3) Budgeted decoding

python - << 'PY'
from src.decoding.budget import BudgetDecoder
dec = BudgetDecoder("outputs/distill/lora_student")  # or "google/flan-t5-small"
text = open("outputs/hierarchy/global_summary.txt").read()
out, log = dec.generate_with_budget(text, budget_tokens=160, profile="fast")
print(log); print("\n---\n", out)
PY

🔧 Configuration Tips

Models: defaults are google/flan-t5-small for summarization/fusion and google/flan-t5-base for LoRA student. Adjust if VRAM is tight.

Chunking: --max_tokens and --overlap control hierarchical coverage vs. cost.

Budgeted decoding profiles: greedy, fast, quality (beams/length_penalty).

Reproducibility: Hugging Face cache + pinned versions yield stable outputs. For strict reproducibility, fix PYTHONHASHSEED and generation seeds.
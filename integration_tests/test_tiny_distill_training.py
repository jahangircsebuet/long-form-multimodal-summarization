import os

def test_tiny_distill_training(tmp_path):
    # 1) Create tiny distillation dataset
    chunks = ["The council approved renovations.", "Construction begins in September."]
    local_summaries = ["Council approved renovations.", "Construction begins in September."]
    fusion_input = "\n".join(local_summaries)
    global_summary = "Council approved renovations; construction starts in September."

    from src.distill.make_dataset import build_pairs
    ds = build_pairs(chunks, local_summaries, fusion_input, global_summary)
    data_dir = tmp_path / "distill_data"
    ds.save_to_disk(str(data_dir))

    # 2) Train LoRA student for a few steps only
    from subprocess import check_call
    import sys
    out_dir = tmp_path / "distill_out"
    cmd = [
        sys.executable, "-m", "src.distill.train_lora_seq2seq",
        "--model", "google/flan-t5-base",
        "--data", str(data_dir),
        "--out_dir", str(out_dir),
        "--epochs", "1",
        "--bs", "2",
        "--grad_accum", "1",
        "--eval_steps", "5",
        "--max_steps", "8"   # tiny run
    ]
    check_call(cmd)

    # 3) Ensure artifacts exist; try a decode with the trained adapter
    assert (out_dir / "lora_student").exists()

    from src.decoding.budget import BudgetDecoder
    dec = BudgetDecoder(str(out_dir / "lora_student"))
    text = "Summarize: The council approved renovations and work begins in September."
    out, log = dec.generate_with_budget(text, budget_tokens=64, profile="greedy")
    assert isinstance(out, str) and len(out) > 0

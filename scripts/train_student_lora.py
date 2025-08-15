import argparse, os, subprocess, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="outputs/hierarchy/distill_data")
    ap.add_argument("--out_dir", default="outputs/distill")
    ap.add_argument("--model", default="google/flan-t5-base")
    ap.add_argument("--max_steps", type=int, default=200)
    args = ap.parse_args()

    cmd = [
        sys.executable, "-m", "src.distill.train_lora_seq2seq",
        "--model", args.model,
        "--data", args.data_dir,
        "--out_dir", args.out_dir,
        "--epochs", "1",
        "--bs", "2",
        "--grad_accum", "4",
        "--eval_steps", "50",
        "--max_steps", str(args.max_steps)
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)

if __name__ == "__main__":
    main()

import argparse, os
from src.segmentation.chunk import chunk_text
from src.teacher.local_summarizer import summarize_chunk, dedupe_summaries
from src.teacher.fuse import fuse_summaries
from src.distill.make_dataset import build_pairs
from datasets import DatasetDict

def main(args):
    text = open(args.input, "r", encoding="utf-8").read()
    chunks = chunk_text(text, max_tokens=args.max_tokens, overlap=args.overlap)
    local = [summarize_chunk(c, max_new_tokens=args.local_len) for c in chunks]
    local = dedupe_summaries(local, sim_thr=args.dedupe_thr)
    fused = fuse_summaries(local, max_new_tokens=args.global_len)

    os.makedirs(args.out_dir, exist_ok=True)
    open(os.path.join(args.out_dir, "global_summary.txt"), "w", encoding="utf-8").write(fused)
    open(os.path.join(args.out_dir, "local_summaries.txt"), "w", encoding="utf-8").write("\n".join(local))

    ds = build_pairs(chunks, local, "\n".join(local), fused)
    ds.save_to_disk(os.path.join(args.out_dir, "distill_data"))
    print(f"Saved fused summary + distill dataset under {args.out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out_dir", default="outputs/hierarchy")
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--overlap", type=int, default=128)
    ap.add_argument("--local_len", type=int, default=128)
    ap.add_argument("--global_len", type=int, default=200)
    ap.add_argument("--dedupe_thr", type=float, default=0.92)
    args = ap.parse_args()
    main(args)

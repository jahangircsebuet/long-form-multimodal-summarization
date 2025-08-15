from pathlib import Path

def test_hierarchy_end2end(tmp_path):
    text = (
        "Post 1: The council approved park renovations and scheduled construction for September. "
        "Post 2: A public forum will occur next week to gather feedback from residents. "
        "Post 3: Funding includes accessibility upgrades and new lighting."
    )
    inp = tmp_path / "thread.txt"
    inp.write_text(text, encoding="utf-8")

    from src.segmentation.chunk import chunk_text
    chunks = chunk_text(text, max_tokens=40, overlap=10)
    assert len(chunks) >= 2

    from src.teacher.local_summarizer import summarize_chunk, dedupe_summaries
    locals_ = [summarize_chunk(c, max_new_tokens=64) for c in chunks]
    locals_ = dedupe_summaries(locals_, sim_thr=0.92)
    assert locals_ and all(isinstance(s, str) for s in locals_)

    from src.teacher.fuse import fuse_summaries
    fused = fuse_summaries(locals_, max_new_tokens=128)
    assert isinstance(fused, str) and len(fused.split()) > 10

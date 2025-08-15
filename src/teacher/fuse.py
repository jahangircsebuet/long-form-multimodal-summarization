from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

_FUSER_MODEL = "google/flan-t5-small"
_tok = None
_model = None

def _ensure():
    global _tok, _model
    if _tok is None or _model is None:
        _tok = AutoTokenizer.from_pretrained(_FUSER_MODEL)
        _model = AutoModelForSeq2SeqLM.from_pretrained(
            _FUSER_MODEL, torch_dtype=torch.float16 if torch.cuda.is_available() else None,
            device_map="auto"
        )

def fuse_summaries(summaries, max_new_tokens: int = 200) -> str:
    _ensure()
    bullets = "\n".join(f"- {s}" for s in summaries if s.strip())
    prompt = (
        "Write a coherent overall summary from these bullet summaries. "
        "Avoid repetition; keep 6-10 sentences; preserve late-document facts; ensure factual consistency.\n\n"
        f"{bullets}"
    )
    inp = _tok(prompt, return_tensors="pt", truncation=True).to(_model.device)
    out = _model.generate(**inp, max_new_tokens=max_new_tokens, num_beams=4)
    return _tok.decode(out[0], skip_special_tokens=True)

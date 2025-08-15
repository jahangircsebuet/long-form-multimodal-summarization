from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
from typing import List
import torch

# small, fast defaults (swap to flan-t5-large for higher quality)
_LOCAL_MODEL = "google/flan-t5-small"
_tok = None
_model = None
_emb = None

def _ensure_models():
    global _tok, _model, _emb
    if _tok is None or _model is None:
        _tok = AutoTokenizer.from_pretrained(_LOCAL_MODEL)
        _model = AutoModelForSeq2SeqLM.from_pretrained(
            _LOCAL_MODEL, torch_dtype=torch.float16 if torch.cuda.is_available() else None,
            device_map="auto"
        )
    if _emb is None:
        _emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def summarize_chunk(chunk: str, max_new_tokens: int = 128) -> str:
    _ensure_models()
    prompt = f"Summarize the following content in 2-3 sentences. Be precise and factual.\n\n{chunk}"
    inp = _tok(prompt, return_tensors="pt", truncation=True).to(_model.device)
    out = _model.generate(**inp, max_new_tokens=max_new_tokens, num_beams=4)
    return _tok.decode(out[0], skip_special_tokens=True)

def dedupe_summaries(summaries: List[str], sim_thr: float = 0.88) -> List[str]:
    """
    Remove highly similar local summaries to keep the fused input compact.
    """
    if not summaries:
        return []
    _ensure_models()
    embs = _emb.encode(summaries, convert_to_tensor=True, normalize_embeddings=True)
    keep_idx = []
    for i in range(len(summaries)):
        dup = False
        for j in keep_idx:
            if float(util.cos_sim(embs[i], embs[j])) >= sim_thr:
                dup = True
                break
        if not dup:
            keep_idx.append(i)
    return [summaries[i] for i in keep_idx]

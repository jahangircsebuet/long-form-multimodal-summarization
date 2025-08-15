from typing import Dict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch, time

_PROFILES = {
    "greedy": dict(num_beams=1, length_penalty=1.0),
    "fast":   dict(num_beams=2, length_penalty=0.9),
    "quality":dict(num_beams=4, length_penalty=1.1),
}

class BudgetDecoder:
    def __init__(self, model_dir_or_id: str = "google/flan-t5-small"):
        self.tok = AutoTokenizer.from_pretrained(model_dir_or_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_dir_or_id, device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else None
        )

    def generate_with_budget(self, text: str, budget_tokens=256, profile="fast"):
        params = _PROFILES.get(profile, _PROFILES["fast"])
        t0 = time.time()
        inp = self.tok(text, return_tensors="pt", truncation=True).to(self.model.device)
        out = self.model.generate(
            **inp, max_new_tokens=budget_tokens,
            num_beams=params["num_beams"], length_penalty=params["length_penalty"],
            no_repeat_ngram_size=3, early_stopping=True
        )
        sec = time.time() - t0
        return self.tok.decode(out[0], skip_special_tokens=True), {
            "latency_s": round(sec, 3),
            "budget_tokens": budget_tokens,
            **params
        }

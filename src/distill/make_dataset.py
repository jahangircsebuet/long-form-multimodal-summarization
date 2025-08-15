from datasets import Dataset, DatasetDict
from typing import List

def build_pairs(chunks: List[str], local_summaries: List[str], fusion_input: str, global_summary: str) -> DatasetDict:
    """
    Produce a tiny distillation dataset:
      - local pairs: chunk -> local_summary
      - one global pair: fused-bullets -> global_summary
    """
    local_pairs = [{"input": c, "target": s} for c, s in zip(chunks, local_summaries)]
    global_pair = [{"input": fusion_input, "target": global_summary}]
    train = Dataset.from_list(local_pairs + global_pair)
    # small dev for sanity
    dev = Dataset.from_list(local_pairs[: max(1, len(local_pairs)//10) ]) if local_pairs else Dataset.from_list(global_pair)
    return DatasetDict({"train": train, "dev": dev, "test": Dataset.from_list([])})

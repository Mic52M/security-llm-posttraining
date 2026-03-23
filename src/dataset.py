from datasets import load_dataset
from typing import Dict

PROMPT_HEADER = "### Istruzione:\n"
RESPONSE_HEADER = "\n\n### Risposta:\n"

def load_security_dataset(train_path: str = "data/train.jsonl",
                          val_path: str = "data/val.jsonl"):
    data_files = {"train": train_path, "validation": val_path}
    ds = load_dataset("json", data_files=data_files)
    return ds

def format_example(example: Dict) -> Dict:
    """
    Converte (input, output) in un singolo campo 'text' per SFT.
    """
    instruction = example["input"]
    answer = example["output"]
    example["text"] = f"{PROMPT_HEADER}{instruction}{RESPONSE_HEADER}{answer}"
    return example

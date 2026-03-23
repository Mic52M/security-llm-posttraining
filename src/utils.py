from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_BASE = "mistralai/Mistral-7B-Instruct-v0.3"
FINETUNED_DIR = "outputs/mistral7b_security_sft"

def load_finetuned_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        FINETUNED_DIR,
        device_map="auto",
        torch_dtype="auto"
    )
    return tokenizer, model

@torch.inference_mode()
def generate_security_answer(prompt: str, max_new_tokens: int = 256) -> str:
    tokenizer, model = load_finetuned_model()
    full_prompt = f"### Istruzione:\n{prompt}\n\n### Risposta:\n"
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return text.split("### Risposta:\n")[-1].strip()

import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from dataset import load_security_dataset, format_example  # stesso folder src/

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

def main():
    # 1. Carica dataset
    dataset = load_security_dataset()
    dataset = dataset.map(format_example)

    # 2. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Modello base
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype="auto"
    )

    # 4. Config LoRA
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    # 5. Config training
    training_args = SFTConfig(
        output_dir="outputs/mistral7b_security_sft",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        max_seq_length=2048,
        dataset_text_field="text",
    )

    # 6. Trainer TRL
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    # 7. Train
    trainer.train()

    # 8. Salvataggio adapter
    os.makedirs("outputs/mistral7b_security_sft", exist_ok=True)
    trainer.save_model("outputs/mistral7b_security_sft")

if __name__ == "__main__":
    main()

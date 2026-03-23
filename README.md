# Security-aligned Mistral 7B (SFT + LoRA)

Questo progetto esplora il fine-tuning supervisionato (SFT) di `mistralai/Mistral-7B-Instruct-v0.3` su un dataset di cybersecurity in lingua italiana, con enfasi su:

- spiegazioni difensive (web security, network security, secure coding),
- analisi di vulnerabilità,
- rifiuto o riscrittura di richieste malevole in ottica etica.

La pipeline usa Hugging Face Transformers, TRL (SFTTrainer) e PEFT/LoRA.

## Struttura

- `data/`: dataset in formato JSONL (`train.jsonl`, `val.jsonl`).
- `src/dataset.py`: caricamento e formattazione esempi.
- `src/train_sft.py`: script di training SFT con LoRA.
- `src/utils.py`: helper per generare risposte con il modello fine-tunato.
- `configs/sft_mistral7b_lora.yaml`: iperparametri principali.

## Esecuzione

```bash
pip install -r requirements.txt

# Primo test di training
python -m src.train_sft
```

In una fase successiva verranno aggiunti:
- una suite di evaluation (capability + safety),
- tecniche di post-training avanzato (es. DPO).

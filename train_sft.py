# ~/medfaith/scripts/train_sft.py
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer
from peft import LoraConfig
from datasets import load_dataset
import yaml, argparse, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default="configs/sft.yaml")
    ap.add_argument("--data", type=str, required=True)
    args = ap.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)

    model_id = cfg["model_id"]
    out_dir  = cfg["output_dir"]

    print("Loading base model:", model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)

    lora_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    ds = load_dataset("json", data_files={"train": args.data})
    trainer = SFTTrainer(
        model=model,
        train_dataset=ds["train"],
        peft_config=lora_config,
        tokenizer=tokenizer,
        args={
            "output_dir": out_dir,
            "per_device_train_batch_size": cfg["per_device_train_batch_size"],
            "gradient_accumulation_steps": cfg["gradient_accumulation_steps"],
            "learning_rate": cfg["lr"],
            "warmup_ratio": cfg["warmup_ratio"],
            "logging_steps": 50,
            "save_steps": cfg["save_every_steps"],
            "bf16": cfg["bf16"],
            "max_steps": cfg["num_train_steps"],
            "max_seq_length": cfg["max_seq_len"],
        },
    )
    trainer.train()
    trainer.save_model(out_dir)
    print("LoRA SFT finished. Checkpoints in", out_dir)

if __name__ == "__main__":
    main()

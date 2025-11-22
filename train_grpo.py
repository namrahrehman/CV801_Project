# ~/medfaith/scripts/train_grpo.py
import os, json, math, argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import PPOTrainer, PPOConfig
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from src.rewards.faithfulness_reward import compute_reward
from src.utils.prompting import make_prompt

def load_records(jsonl_path):
    recs = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))
    return recs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)                   # e.g. Qwen2-VL-2B
    ap.add_argument("--sft_ckpt", required=True)                     # e.g. checkpoints/qwen2vl_sft
    ap.add_argument("--data_jsonl", required=True)                   # JSONL with {image_path, question, answer, ...}
    ap.add_argument("--out_dir", default="checkpoints/qwen2vl_grpo")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--weights", type=str, default="0.2,0.4,0.2,0.2") # fmt,ans,cons,cf
    args = ap.parse_args()

    w_fmt, w_ans, w_cons, w_cf = [float(x) for x in args.weights.split(",")]
    weights = {"fmt": w_fmt, "ans": w_ans, "cons": w_cons, "cf": w_cf}

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base, args.sft_ckpt)           # start GRPO from SFT checkpoint
    model.eval()

    # sentence-transformer for s(Q,C)
    biobert = SentenceTransformer("pritamdeka/BioBERT-mnli")

    # PPO config
    ppo_config = PPOConfig(
        steps=args.steps,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        optimize_cuda_cache=True,
        log_with=None,
        save_freq=200
    )
    trainer = PPOTrainer(ppo_config, model, tokenizer)

    # load supervision records produced by your evaluation harness
    # expected fields per record:
    # { "image_path": "...", "question": "...", "gold": "...",
    #   "valid_json": bool, "answer_correct": bool,
    #   "caption": "...", "delta_cf_roi": float, "delta_cf_rand": float }
    records = load_records(args.data_jsonl)

    # collate into batches of questions and optional image paths
    def chunk(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i+n]

    os.makedirs(args.out_dir, exist_ok=True)

    step = 0
    for batch in chunk(records, ppo_config.batch_size):
        step += 1

        # build prompts for generation
        prompts = []
        metas = []
        for rec in batch:
            q = rec.get("question", "")
            img_path = rec.get("image_path", None)                    # keep for future use if you wire VLM pixels
            prompts.append(make_prompt(q))                            # JSON schema system+user prompt
            metas.append(rec)

        # tokenize prompts
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

        # generate responses
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        gen_texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        # parse structured JSON from texts using your same helper
        # here we inline a light parser to keep the script self-contained
        def parse_json_safe(txt):
            import re, json as _json
            m = re.search(r"\{.*\}", txt, re.DOTALL)
            if not m: return {}
            try: return _json.loads(m.group(0))
            except: return {}

        # compute rewards per sample
        rewards = []
        for rec, txt in zip(metas, gen_texts):
            js = parse_json_safe(txt)
            # create a synthetic record with fields expected by compute_reward
            r_for_reward = {
                "valid_json": isinstance(js, dict) and len(js) > 0,
                "answer_correct": str(js.get("answer","")).strip().lower()[:1] == str(rec.get("gold","")).strip().lower()[:1],
                "question": rec.get("question",""),
                "caption": js.get("caption",""),
                "delta_cf_roi": rec.get("delta_cf_roi", 0.0),
                "delta_cf_rand": rec.get("delta_cf_rand", 0.0)
            }
            # overlap helper for s(Q,C)
            def overlap_func(q, c):
                import re
                ORG = {"lung","heart","liver","kidney","skull","brain","spine","rib","femur","arm","chest"}
                SIDE = {"left","right","bilateral","unilateral"}
                FIND = {"mass","nodule","lesion","fracture","opacity","effusion","consolidation","atelectasis","pneumothorax"}
                def norm(s): return re.sub(r"[^a-z0-9 ]","", s.lower())
                qn, cn = norm(q), norm(c)
                terms = [t for t in ORG|SIDE|FIND if t in qn]
                if not terms: return 0.0
                return sum(1 for t in terms if t in cn) / len(terms)

            R = compute_reward(r_for_reward, weights, biobert, overlap_func)
            rewards.append(R)

        # convert to tensors
        rewards_t = [torch.tensor(r, dtype=torch.float32, device=model.device) for r in rewards]

        # PPO step requires query tensors and response tensors
        # TRL expects "queries" = input_ids, "responses" = generated ids minus prompt part
        queries = [inputs.input_ids[i] for i in range(inputs.input_ids.size(0))]
        responses = []
        for i in range(len(gen_ids)):
            # cut off the prompt tokens to keep only generated continuation
            prompt_len = inputs.input_ids.size(1)
            responses.append(gen_ids[i][prompt_len:])

        trainer.step(queries=queries, responses=responses, rewards=rewards_t)

        if step % 50 == 0:
            print(f"Step {step} avg reward {float(torch.stack(rewards_t).mean()):.3f}")

        if step % 200 == 0:
            save_path = os.path.join(args.out_dir, f"step_{step}")
            trainer.save_pretrained(save_path)

    trainer.save_pretrained(args.out_dir)
    print("GRPO-lite finished, saved to", args.out_dir)

if __name__ == "__main__":
    main()

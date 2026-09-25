"""GRPO with TRL on a rented cloud GPU (Phase 5). Do not run on the Mac (see --dry-run).

    pip install -r requirements-cloud.txt
    python sql_env.py download
    python train_cloud.py                                              # 0.5B, one epoch over ~6,000 questions
    python train_cloud.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct --lora-rank 16 --wandb --use-vllm

Same reward (sql_env.shaped_reward), same data split (load_rl_train: 10 validation databases
held out, dev untouched) and the same defaults as the validated local run 4 in BERICHT.md:
lr 1e-6 (constant), KL beta 0.04, 8 answers per question, temperature 0.8, top-p 0.95, top-k 20,
LoRA rank 8 on all linear layers with scale 20 (MLX `scale` = PEFT `lora_alpha / r`).
TRL's default loss "dapo" averages over all answer tokens of the batch, like our local loss.

Writes to --output-dir:
  adapter/        PEFT LoRA adapter
  merged/         base model with LoRA merged; evaluate on the Mac with
                  python evaluate.py --model <merged> --split val
  steps.json      per-batch log in the local format; plot with python train_local.py --plot steps.json
  log_history.json  TRL's own logs (loss, KL, rewards, entropy, ...)
"""
import argparse
import json
import random
import statistics
import time
from pathlib import Path

from sql_env import REWARD_VARIANTS, ROOT, RewardLog, build_prompt, load_rl_train, within_question_trend

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]  # = MLX default


def completion_text(c):
    """TRL passes conversational completions as [{"role": "assistant", "content": ...}]."""
    return c[0]["content"] if isinstance(c, list) else c


def make_reward_func(log):
    """TRL reward function: extra dataset columns (db_id, query, question) arrive as lists."""

    def reward_func(completions, db_id, query, question, log_metric=None, **_):
        examples = [{"db_id": d, "query": q, "question": qs} for d, q, qs in zip(db_id, query, question)]
        rewards, row = log.grade([completion_text(c) for c in completions], examples)
        if log_metric is not None:
            for key in ("accuracy", "no_signal_share", "sql_block_share", "mean_chars"):
                log_metric(f"sql/{key}", row[key])
        return rewards

    reward_func.__name__ = f"sql_{log.variant}"  # TRL names its reward metrics after this
    return reward_func


def build_dataset(tokenizer, n_train, max_prompt_tokens, seed):
    """Conversational prompts (system + schema/question) plus the columns the reward needs."""
    from datasets import Dataset

    pool = load_rl_train()
    random.Random(seed).shuffle(pool)
    rows, skipped = [], 0
    for ex in pool:
        messages = build_prompt(ex)
        if max_prompt_tokens:
            text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            if len(tokenizer.encode(text)) > max_prompt_tokens:
                skipped += 1
                continue
        rows.append({"prompt": messages, "db_id": ex["db_id"], "query": ex["query"], "question": ex["question"]})
        if n_train and len(rows) == n_train:
            break
    return Dataset.from_list(rows), skipped


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    ap.add_argument("--reward", default="binary", choices=REWARD_VARIANTS)
    ap.add_argument("--num-generations", type=int, default=8, help="answers per question (GRPO group size)")
    ap.add_argument("--prompts-per-step", type=int, default=8, help="questions per optimizer step")
    ap.add_argument("--grad-accum", type=int, default=1, help="split each optimizer step into micro-batches (memory)")
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--max-steps", type=int, default=0, help="optimizer steps; 0 = one epoch")
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--lora-scale", type=float, default=20.0, help="lora_alpha = scale * rank (MLX convention)")
    ap.add_argument("--beta", type=float, default=0.04, help="KL penalty to the base model")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-completion-length", type=int, default=256)
    ap.add_argument("--max-prompt-tokens", type=int, default=1024, help="skip longer training prompts; 0 = keep all")
    ap.add_argument("--n-train", type=int, default=0, help="subset of training questions; 0 = all")
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", default=None, help="default checkpoints/cloud_<model>_<reward>")
    ap.add_argument("--wandb", action="store_true", help="log to Weights & Biases (run `wandb login` first)")
    ap.add_argument("--use-vllm", action="store_true", help="fast generation with vLLM on the same GPU")
    ap.add_argument("--dry-run", action="store_true",
                    help="build data, reward, config and trainer, grade gold answers, then stop without training")
    args = ap.parse_args()

    import torch
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    completions_per_step = args.prompts_per_step * args.num_generations
    if completions_per_step % args.grad_accum:
        ap.error("prompts-per-step * num-generations must be divisible by grad-accum")
    out = Path(args.output_dir or ROOT / "checkpoints" / f"cloud_{args.model.split('/')[-1]}_{args.reward}")
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataset, skipped = build_dataset(tokenizer, 16 if args.dry_run else args.n_train, args.max_prompt_tokens, args.seed)
    log = RewardLog(args.reward, args.num_generations)
    reward_func = make_reward_func(log)

    cuda = torch.cuda.is_available()
    config = GRPOConfig(
        output_dir=str(out),
        learning_rate=args.lr,
        lr_scheduler_type="constant",  # as in the local MLX run
        num_generations=args.num_generations,
        per_device_train_batch_size=completions_per_step // args.grad_accum,  # counted in completions
        gradient_accumulation_steps=args.grad_accum,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=0.95,
        top_k=20,
        beta=args.beta,
        loss_type="dapo",
        scale_rewards="group",
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        num_train_epochs=1,
        gradient_checkpointing=True,
        bf16=cuda and torch.cuda.is_bf16_supported(),
        # Without CUDA (only --dry-run on a Mac): load on CPU. Loading straight to Apple's MPS
        # device with transformers 5's threaded loader segfaults.
        model_init_kwargs=None if cuda else {"device_map": "cpu"},
        use_vllm=args.use_vllm,
        vllm_mode="colocate",
        vllm_max_model_length=(args.max_prompt_tokens or 2048) + args.max_completion_length,
        logging_steps=1,
        log_completions=True,
        num_completions_to_print=2,  # spot checks against reward hacking
        save_steps=args.save_steps,
        report_to="wandb" if args.wandb else "none",
        run_name=out.name,
        seed=args.seed,
    )
    peft_config = LoraConfig(r=args.lora_rank, lora_alpha=round(args.lora_scale * args.lora_rank), lora_dropout=0.0,
                             target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
    trainer = GRPOTrainer(model=args.model, reward_funcs=reward_func, args=config, train_dataset=dataset,
                          processing_class=tokenizer, peft_config=peft_config)

    if args.dry_run:
        gold = [[{"role": "assistant", "content": f"```sql\n{q}\n```"}] for q in dataset["query"]]
        rewards = reward_func(gold, dataset["db_id"], dataset["query"], dataset["question"])
        log.steps.clear()
        trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        print(f"dry run ok: {len(dataset)} prompts, gold answers score {statistics.mean(rewards):.2f} (must be 1.00), "
              f"{trainable:,} trainable LoRA parameters, {completions_per_step} completions per step, "
              f"skipped {skipped} long prompts. No training.")
        return

    t0 = time.perf_counter()
    trainer.train()
    minutes = (time.perf_counter() - t0) / 60

    trainer.save_model(str(out / "adapter"))
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(str(out / "merged"))
    tokenizer.save_pretrained(str(out / "merged"))

    k = min(50, len(log.steps) // 2)
    summary = {
        "model": args.model, "reward": args.reward, "train_questions": len(dataset), "skipped_long_prompts": skipped,
        "num_generations": args.num_generations, "prompts_per_step": args.prompts_per_step, "grad_accum": args.grad_accum,
        "lr": args.lr, "beta": args.beta, "lora_rank": args.lora_rank, "lora_alpha": round(args.lora_scale * args.lora_rank),
        "temperature": args.temperature, "max_completion_length": args.max_completion_length,
        "optimizer_steps": trainer.state.global_step, "logged_batches": len(log.steps), "minutes": round(minutes, 1),
        "gpu": torch.cuda.get_device_name(0) if cuda else "none", "vllm": args.use_vllm,
        f"accuracy_first{k}": round(statistics.mean(s["accuracy"] for s in log.steps[:k]), 4) if k else None,
        f"accuracy_last{k}": round(statistics.mean(s["accuracy"] for s in log.steps[-k:]), 4) if k else None,
        "accuracy_trend_within_question": within_question_trend(log.steps),
        "date": time.strftime("%Y-%m-%d"),
    }
    (out / "steps.json").write_text(json.dumps({"summary": summary, "steps": log.steps}, indent=2, ensure_ascii=False))
    (out / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=2))
    print(json.dumps(summary, indent=2))
    print("written:", out)


if __name__ == "__main__":
    main()

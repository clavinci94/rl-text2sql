"""GRPO smoke test on the Mac (MLX + mlx-lm-lora), Phase 4.

    python train_local.py                          # 200 train questions, 8 answers each, 300 steps (~70 min)
    python train_local.py --reward length --iters 50

Only goal: show that the mean reward rises. Writes
  checkpoints/<run>/            LoRA adapter (python evaluate.py --adapter checkpoints/<run>)
  results/<run>.json            config, per-step log, timing, full-run estimate
  results/<run>_reward.png      reward curve
"""
import argparse
import json
import math
import random
import statistics
import time
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from mlx_lm import load
from mlx_lm.tuner.callbacks import TrainingCallback
from mlx_lm.tuner.utils import linear_to_lora_layers
from mlx_lm_lora.trainer.datasets import GRPODataset
from mlx.utils import tree_flatten
from mlx_lm_lora.trainer.grpo_trainer import GRPOTrainingArgs, train_grpo

from sql_env import REWARD_VARIANTS, ROOT, RewardLog, build_prompt, load_rl_train, within_question_trend


class SQLReward:
    """mlx-lm-lora reward function around sql_env.RewardLog.

    mlx-lm-lora 3.1.3 calls the reward function twice per step with identical inputs
    (rewards, then metrics). Only that immediate second call is served from cache: no
    second SQL execution, no second log row. (An earlier version cached by content alone
    and skipped a step when a collapsed model repeated the same answers to the same batch.)"""

    def __init__(self, variant, group_size):
        self.log = RewardLog(variant, group_size)
        self.steps = self.log.steps
        self._pending = None  # (key, rewards) of a first call still waiting for its repeat
        self.__name__ = f"sql_{variant}"  # the trainer names its metrics after this

    def __call__(self, prompts, completions, answer, types=None):
        key = (tuple(completions), tuple(answer))
        if self._pending is not None and self._pending[0] == key:
            rewards, self._pending = self._pending[1], None
            return rewards
        rewards, _ = self.log.grade(completions, [json.loads(a) for a in answer])
        self._pending = (key, rewards)
        return rewards


def completion_logps(model, prompts, completions, batch_indices):
    """Per-token log-probs of each completion given its prompt, shape (N, max_len), plus mask.

    Only the completion positions go through the output layer, which keeps memory low
    for long schema prompts. Uses shapes only (no array evaluation): the trainer
    compiles its step function."""
    seqs, spans = [], []
    for comp, b in zip(completions, batch_indices):
        p = mx.array(prompts[b], dtype=mx.int32)
        seqs.append(mx.concatenate([p, comp.astype(mx.int32)]))
        spans.append((p.shape[0], comp.shape[0]))
    width = max(s.shape[0] for s in seqs)
    inputs = mx.stack([mx.concatenate([s, mx.zeros((width - s.shape[0],), mx.int32)]) for s in seqs])
    hidden = model.model(inputs)
    head = model.lm_head if hasattr(model, "lm_head") else model.model.embed_tokens.as_linear
    c_max = max(c for _, c in spans)
    rows, masks = [], []
    for i, (p, c) in enumerate(spans):
        logits = head(hidden[i, p - 1:p - 1 + c]).astype(mx.float32)  # position t predicts token t+1
        logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        lp = mx.take_along_axis(logp, inputs[i, p:p + c][:, None], axis=-1).squeeze(-1)
        rows.append(mx.concatenate([lp, mx.zeros((c_max - c,))]))
        masks.append(mx.concatenate([mx.ones((c,)), mx.zeros((c_max - c,))]))
    return mx.stack(rows), mx.stack(masks)


def make_grpo_loss(eos_id):
    """GRPO loss conditioned on the prompt.

    mlx-lm-lora 3.1.3's grpo_loss scores the completion tokens *without* the prompt
    (schema + question), so it reinforces SQL text regardless of the question. Our
    first 300-step run degraded (accuracy 0.45 -> 0.29) with it. This replacement
    conditions on the prompt, scores the first completion token, and appends EOS to
    completions that stopped on their own, so the model also learns when to stop.
    One update per sampled batch: the importance ratio is 1, so no clipping is needed.
    Normalization matches the library's default: mean over all completion tokens."""

    def loss_fn(model, batch=None, completions=None, batch_indices=None, advantages=None,
                reward_metrics=None, beta=0.0, ref_model=None, max_tokens=256, **_):
        prompts = batch[0]
        comps = [c if c.shape[0] >= max_tokens else mx.concatenate([c.astype(mx.int32), mx.array([eos_id], mx.int32)])
                 for c in completions]
        lp, mask = completion_logps(model, prompts, comps, batch_indices)
        adv = advantages.reshape(-1, 1)
        per_token = -mx.exp(lp - mx.stop_gradient(lp)) * adv
        if ref_model is not None:
            ref_lp = mx.stop_gradient(completion_logps(ref_model, prompts, comps, batch_indices)[0])
            kl = mx.exp(ref_lp - lp) - (ref_lp - lp) - 1  # Schulman's unbiased estimator
        else:
            kl = mx.zeros_like(lp)
        per_token = per_token + beta * kl
        n_tokens = mask.sum()
        loss = (per_token * mask).sum() / n_tokens
        lengths = [c.shape[0] for c in completions]
        metrics = {
            "kl": ((kl * mask).sum(axis=1) / mx.maximum(mask.sum(axis=1), 1)).mean(),
            "average_generated_tokens": sum(lengths) / len(lengths),
            "max_generated_tokens": max(lengths), "min_generated_tokens": min(lengths),
            "hit_max_tokens_ratio": sum(n >= max_tokens for n in lengths) / len(lengths),
            "clip_ratio_low": mx.array(0.0), "clip_ratio_high": mx.array(0.0), "clip_ratio_total": mx.array(0.0),
            **reward_metrics,
        }
        return loss, n_tokens, metrics

    return loss_fn


class StepTimer(TrainingCallback):
    def __init__(self, reward_log):
        self.reward_log, self.t = reward_log, time.perf_counter()

    def on_train_loss_report(self, info):
        now = time.perf_counter()
        row = self.reward_log.steps[info["iteration"] - 1]
        row.update(step_s=round(now - self.t, 2), loss=info["train_loss"], kl=info.get("train_kl"),
                   gen_tokens=info.get("train_average_generated_tokens"), peak_mem_gb=round(info["peak_memory"], 2))
        self.t = now
        if row["step"] % 10 == 0:
            recent = self.reward_log.steps[-10:]
            print(f"step {row['step']:4d}  reward(last10) {statistics.mean(s['reward_mean'] for s in recent):.3f}"
                  f"  acc(last10) {statistics.mean(s['accuracy'] for s in recent):.3f}"
                  f"  {row['step_s']:.1f}s/step", flush=True)


def moving_average(values, window):
    out = []
    for i in range(len(values)):
        chunk = values[max(0, i - window + 1):i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def plot(steps, title, path, window=20):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, muted, grid, blue = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"
    x = [s["step"] for s in steps]
    panels = [("reward_mean", "Mittlerer Reward pro Schritt"),
              ("accuracy", "Anteil korrekter Antworten"),
              ("no_signal_share", "Anteil Gruppen ohne Lernsignal (alle gleich bewertet)")]
    if all(s["reward_mean"] == s["accuracy"] for s in steps):  # binary reward: same curve twice
        panels = [("accuracy", "Mittlerer Reward pro Schritt (binär, = Anteil korrekter Antworten)"), panels[2]]
    fig, axes = plt.subplots(len(panels), 1, figsize=(9, 2.9 * len(panels)), sharex=True, facecolor=surface)
    for ax, (key, label) in zip(axes, panels):
        y = [s[key] for s in steps]
        ma = moving_average(y, window)
        ax.set_facecolor(surface)
        ax.plot(x, y, color=blue, alpha=0.3, linewidth=1)
        ax.plot(x, ma, color=blue, linewidth=2)
        ax.set_title(f"{label}\ndünn: pro Schritt · dick: gleitender Mittelwert über {window} Schritte "
                     f"(Start {ma[min(window, len(ma)) - 1]:.2f}, Ende {ma[-1]:.2f})",
                     loc="left", fontsize=10, color=ink)
        ax.grid(axis="y", color=grid, linewidth=0.8)
        ax.tick_params(colors=muted, labelsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
    axes[-1].set_xlabel("Trainingsschritt", color=muted)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=ink)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    ap.add_argument("--reward", default="binary", choices=REWARD_VARIANTS)
    ap.add_argument("--n-train", type=int, default=200, help="random subset of the filtered RL train set")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--batch", type=int, default=2, help="questions per step")
    # Defaults = the setting that worked (run 4 in BERICHT.md). lr 1e-5 without KL destroyed
    # the output format within 5 steps; with 4 answers most groups carried no learning signal.
    ap.add_argument("--group", type=int, default=8, help="answers per question")
    ap.add_argument("--beta", type=float, default=0.04, help="KL penalty to the base model (0 = no reference model)")
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--lora-scale", type=float, default=20.0)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--max-prompt-tokens", type=int, default=1024, help="skip longer training prompts (memory)")
    ap.add_argument("--no-grad-checkpoint", action="store_true", help="faster, needs more memory")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run", default=None, help="run name, default grpo_local_<reward>")
    ap.add_argument("--plot", metavar="STEPS_JSON", default=None,
                    help="only draw the reward curve of an existing log (e.g. steps.json of a cloud run) and exit")
    args = ap.parse_args()
    if args.plot:
        log = json.loads(Path(args.plot).read_text())
        png = Path(args.plot).with_name(Path(args.plot).stem + "_reward.png")
        s = log.get("summary", {})
        plot(log["steps"], f"GRPO, {s.get('model', '?').split('/')[-1]}, Reward «{s.get('reward', '?')}»", png)
        print("written:", png, "| trend:", within_question_trend(log["steps"]))
        return
    run = args.run or f"grpo_local_{args.reward}"

    random.seed(args.seed)
    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    model, tokenizer = load(args.model)

    # The loss runs prompt + answer through the network (with gradients), so very long
    # schemas exhaust 24 GB. Dev prompts are at most 928 tokens, so a 1,024 cap keeps
    # every test-like prompt length.
    rl_train = load_rl_train()
    pool = rl_train[:]
    random.Random(args.seed).shuffle(pool)
    data, n_skipped = [], 0
    for ex in pool:
        messages = build_prompt(ex)
        if len(tokenizer.apply_chat_template(messages, add_generation_prompt=True)) > args.max_prompt_tokens:
            n_skipped += 1
            continue
        data.append({"system": messages[0]["content"], "prompt": messages[1]["content"], "answer": json.dumps(ex)})
        if len(data) == args.n_train:
            break

    ref_model = load(args.model)[0] if args.beta > 0 else None
    model.freeze()
    lora = {"rank": args.lora_rank, "scale": args.lora_scale, "dropout": 0.0}
    linear_to_lora_layers(model, len(model.layers), lora)
    model.train()
    n_trainable = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))

    ckpt = ROOT / "checkpoints" / run
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "adapter_config.json").write_text(json.dumps(
        {"fine_tune_type": "lora", "num_layers": len(model.layers), "lora_parameters": lora,
         "base_model": args.model}, indent=2))

    reward_fn = SQLReward(args.reward, args.group)
    targs = GRPOTrainingArgs(
        batch_size=args.batch, iters=args.iters, group_size=args.group, beta=args.beta,
        max_completion_length=args.max_tokens, temperature=args.temperature,
        steps_per_report=1, steps_per_eval=10**9, steps_per_save=100,
        max_seq_length=2304, adapter_file=str(ckpt / "adapters.safetensors"),
        grad_checkpoint=not args.no_grad_checkpoint)

    t0 = time.perf_counter()
    train_grpo(model, ref_model, tokenizer, optim.Adam(learning_rate=args.lr), GRPODataset(data, tokenizer),
               None, reward_funcs=[reward_fn], args=targs, training_callback=StepTimer(reward_fn),
               loss_fn=make_grpo_loss(tokenizer.eos_token_id), end_answer_token="")
    total_s = time.perf_counter() - t0

    steps = reward_fn.steps
    step_times = [s["step_s"] for s in steps[1:] if "step_s" in s]  # step 1 includes compilation
    median_step = statistics.median(step_times)
    k = min(50, len(steps) // 2)
    first, last = steps[:k], steps[-k:]
    full_epoch_steps = math.ceil(len(rl_train) / args.batch)
    summary = {
        "run": run, "model": args.model, "reward": args.reward, "n_train": args.n_train,
        "iters": args.iters, "batch": args.batch, "group": args.group, "beta": args.beta, "lr": args.lr,
        "lora": lora, "trainable_params": n_trainable, "temperature": args.temperature,
        "max_tokens": args.max_tokens, "seed": args.seed, "loss": "prompt-conditioned (train_local.make_grpo_loss)",
        "max_prompt_tokens": args.max_prompt_tokens, "skipped_long_prompts_while_sampling": n_skipped,
        "grad_checkpoint": not args.no_grad_checkpoint,
        f"reward_first{k}": round(statistics.mean(s["reward_mean"] for s in first), 4),
        f"reward_last{k}": round(statistics.mean(s["reward_mean"] for s in last), 4),
        f"accuracy_first{k}": round(statistics.mean(s["accuracy"] for s in first), 4),
        f"accuracy_last{k}": round(statistics.mean(s["accuracy"] for s in last), 4),
        "no_signal_share": round(statistics.mean(s["no_signal_share"] for s in steps), 4),
        f"sql_block_share_first{k}": round(statistics.mean(s["sql_block_share"] for s in first), 4),
        f"sql_block_share_last{k}": round(statistics.mean(s["sql_block_share"] for s in last), 4),
        "accuracy_trend_within_question": within_question_trend(steps),
        "step_s_median": round(median_step, 2), "step_s_mean": round(statistics.mean(step_times), 2),
        "step_s_first": steps[0].get("step_s"), "total_s": round(total_s, 1),
        "peak_mem_gb": max(s.get("peak_mem_gb", 0) for s in steps),
        "full_epoch_estimate": {"rl_train_questions": len(rl_train), "steps": full_epoch_steps,
                                "hours_at_median_step": round(full_epoch_steps * median_step / 3600, 2)},
        "date": time.strftime("%Y-%m-%d"),
    }
    out = ROOT / "results" / f"{run}.json"
    out.write_text(json.dumps({"summary": summary, "steps": steps}, indent=2, ensure_ascii=False))
    plot(steps, f"GRPO lokal, {args.model.split('/')[-1]}, Reward «{args.reward}», "
                f"{args.n_train} Fragen × {args.group} Antworten", ROOT / "results" / f"{run}_reward.png")
    print(json.dumps(summary, indent=2))
    print("written:", out, "and", out.with_name(f"{run}_reward.png"), "| adapter:", ckpt)


if __name__ == "__main__":
    main()

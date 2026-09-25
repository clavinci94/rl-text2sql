"""Execution accuracy of a model on the held-out Spider dev set (MLX, greedy decoding).

    python evaluate.py --n 200                      # fixed random subset (seed 0)
    python evaluate.py                              # full dev set (1,034)
    python evaluate.py --adapter checkpoints/x --tag grpo   # LoRA model later

Writes results/<tag>_<model>[_n<N>].json with a summary and every model answer.
"""
import argparse
import json
import platform
import random
import statistics
import time
from collections import Counter
from importlib.metadata import version

from mlx_lm import load, stream_generate
from tqdm import tqdm

from sql_env import ROOT, build_prompt, gold_result, load_split, load_val, score


def error_type(result):
    """Split execution errors into syntax / schema / other, using SQLite's message."""
    if result["status"] != "exec_error":
        return result["status"]
    msg = result["error"].lower()
    if any(s in msg for s in ("syntax error", "incomplete input", "unrecognized token")):
        return "syntax_error"
    if any(s in msg for s in ("no such table", "no such column", "ambiguous column")):
        return "schema_error"
    return "other_exec_error"


def generate_one(model, tokenizer, messages, max_tokens):
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    prompt = tokenizer.encode(text)
    t0 = time.perf_counter()
    out, last = "", None
    for last in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens):
        out += last.text
    return out, time.perf_counter() - t0, last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    ap.add_argument("--adapter", default=None, help="LoRA adapter directory (optional)")
    ap.add_argument("--n", type=int, default=0, help="random dev subset size, 0 = all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--split", default="dev", choices=["dev", "val"],
                    help="val = held-out train databases for tuning; dev = final test set only")
    args = ap.parse_args()

    dev = load_split("dev") if args.split == "dev" else load_val()
    idx = list(range(len(dev)))
    if args.n:
        idx = sorted(random.Random(args.seed).sample(idx, args.n))

    model, tokenizer = load(args.model, adapter_path=args.adapter)
    generate_one(model, tokenizer, build_prompt(dev[0]), 16)  # warm-up, not timed

    rows, t_start = [], time.perf_counter()
    bar = tqdm(idx, desc="eval")
    for i in bar:
        ex = dev[i]
        response, latency, last = generate_one(model, tokenizer, build_prompt(ex), args.max_tokens)
        r = score(response, ex)
        rows.append({
            "dev_index": i, "db_id": ex["db_id"], "question": ex["question"], "gold": ex["query"],
            "response": response, "sql": r["sql"], "formatted": r["formatted"],
            "status": error_type(r), "error": r["error"], "correct": r["status"] == "correct",
            "gold_empty": gold_result(ex["db_id"], ex["query"])[0] == [],
            "latency_s": round(latency, 4), "prompt_tokens": last.prompt_tokens,
            "completion_tokens": last.generation_tokens, "finish_reason": last.finish_reason,
            "generation_tps": round(last.generation_tps, 1), "peak_memory_gb": round(last.peak_memory, 3),
        })
        bar.set_postfix(acc=f"{sum(x['correct'] for x in rows) / len(rows):.3f}")

    n = len(rows)
    lat = sorted(x["latency_s"] for x in rows)
    non_empty = [x for x in rows if not x["gold_empty"]]
    summary = {
        "model": args.model, "adapter": args.adapter, "tag": args.tag,
        "split": args.split, "n": n, "subset_seed": args.seed if args.n else None,
        "decoding": "greedy", "max_tokens": args.max_tokens,
        "execution_accuracy": round(sum(x["correct"] for x in rows) / n, 4),
        "execution_accuracy_excl_empty_gold": round(
            sum(x["correct"] for x in non_empty) / max(len(non_empty), 1), 4),
        "n_excl_empty_gold": len(non_empty),
        "sql_block_rate": round(sum(x["formatted"] for x in rows) / n, 4),
        "status_counts": dict(Counter(x["status"] for x in rows).most_common()),
        "latency_s": {"mean": round(statistics.mean(lat), 3), "median": round(statistics.median(lat), 3),
                      "p95": lat[int(0.95 * (n - 1))], "max": lat[-1]},
        "mean_prompt_tokens": round(statistics.mean(x["prompt_tokens"] for x in rows), 1),
        "mean_completion_tokens": round(statistics.mean(x["completion_tokens"] for x in rows), 1),
        "truncated_at_max_tokens": sum(x["finish_reason"] == "length" for x in rows),
        "peak_memory_gb": max(x["peak_memory_gb"] for x in rows),
        "wall_time_s": round(time.perf_counter() - t_start, 1),
        "hardware": f"{platform.machine()} {platform.mac_ver()[0]}", "mlx_lm": version("mlx-lm"),
        "date": time.strftime("%Y-%m-%d"),
    }
    name = (f"{args.tag}_{args.model.rstrip('/').split('/')[-1]}" + (f"_n{args.n}" if args.n else "")
            + ("_val" if args.split == "val" else ""))
    out = ROOT / "results" / f"{name}.json"
    out.write_text(json.dumps({"summary": summary, "examples": rows}, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2))
    print("written:", out)


if __name__ == "__main__":
    main()

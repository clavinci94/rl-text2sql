"""Execution accuracy of a model on the held-out Spider dev set.

Local models run with MLX (greedy decoding); --api-model evaluates a Claude model through
the Anthropic API as the "large API model" baseline, with measured tokens and cost per request.

    python evaluate.py --n 200                      # fixed random subset (seed 0)
    python evaluate.py                              # full dev set (1,034)
    python evaluate.py --adapter checkpoints/x --tag grpo   # LoRA model
    python evaluate.py --api-model claude-opus-5 --split val --dry-run   # show the request, no API call
    python evaluate.py --api-model claude-opus-5 --tag api               # costs money, needs credentials

Writes results/<tag>_<model>[_n<N>][_val].json with a summary and every model answer.
"""
import argparse
import json
import platform
import random
import statistics
import time
from collections import Counter
from importlib.metadata import version

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


# Anthropic first-party prices in USD per 1M tokens (input, output), from the claude-api skill's
# model table (cached 2026-06-24). Check the live pricing page before quoting results.
API_PRICES = {"claude-opus-5": (5.0, 25.0), "claude-opus-4-8": (5.0, 25.0), "claude-opus-5-5": (4.0, 20.0),
              "claude-sonnet-5": (2.0, 10.0), "claude-haiku-4-5": (1.0, 5.0), "claude-fable-5-1": (10.0, 50.0)}


def api_request(model, messages, effort):
    """Same system prompt and user message as the local models. Sampling parameters such as
    temperature are not accepted by current Claude models; thinking stays at its adaptive default.
    Refusal fallbacks are on (server-side, routed by refusal category); served_by records the model."""
    return {"model": model, "max_tokens": 16000, "system": messages[0]["content"],
            "messages": [{"role": "user", "content": messages[1]["content"]}],
            "output_config": {"effort": effort},
            "betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}


def generate_api(client, request):
    import anthropic
    t0 = time.perf_counter()
    try:
        resp = client.beta.messages.create(**request)
    except anthropic.AuthenticationError:
        raise
    except (anthropic.RateLimitError, anthropic.APIStatusError, anthropic.APIConnectionError) as e:
        return "", time.perf_counter() - t0, {"api_error": f"{type(e).__name__}: {e}"}
    latency = time.perf_counter() - t0
    text = "".join(b.text for b in resp.content if b.type == "text")
    price_in, price_out = API_PRICES.get(resp.model, (float("nan"), float("nan")))
    u = resp.usage
    return text, latency, {
        "prompt_tokens": u.input_tokens, "completion_tokens": u.output_tokens,  # output incl. thinking
        "finish_reason": "length" if resp.stop_reason == "max_tokens" else resp.stop_reason,
        "refusal": resp.stop_reason == "refusal", "served_by": resp.model,
        "cost_usd": (u.input_tokens * price_in + u.output_tokens * price_out) / 1e6,
    }


def generate_one(model, tokenizer, messages, max_tokens):
    from mlx_lm import stream_generate
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
    ap.add_argument("--api-model", default=None, help="evaluate a Claude model via the Anthropic API instead")
    ap.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"],
                    help="API only; high is the API default")
    ap.add_argument("--dry-run", action="store_true", help="API only: print the first request, make no call")
    args = ap.parse_args()
    if args.api_model:
        args.model = args.api_model

    dev = load_split("dev") if args.split == "dev" else load_val()
    idx = list(range(len(dev)))
    if args.n:
        idx = sorted(random.Random(args.seed).sample(idx, args.n))

    if args.api_model:
        if args.dry_run:
            print(json.dumps(api_request(args.api_model, build_prompt(dev[idx[0]]), args.effort), indent=2))
            print(f"dry run: {len(idx)} requests would be sent to {args.api_model}. No API call made.")
            return
        import anthropic
        client = anthropic.Anthropic()
    else:
        from mlx_lm import load
        model, tokenizer = load(args.model, adapter_path=args.adapter)
        generate_one(model, tokenizer, build_prompt(dev[0]), 16)  # warm-up, not timed

    rows, t_start = [], time.perf_counter()
    bar = tqdm(idx, desc="eval")
    for i in bar:
        ex = dev[i]
        if args.api_model:
            response, latency, info = generate_api(client, api_request(args.api_model, build_prompt(ex), args.effort))
        else:
            response, latency, last = generate_one(model, tokenizer, build_prompt(ex), args.max_tokens)
            info = {"prompt_tokens": last.prompt_tokens, "completion_tokens": last.generation_tokens,
                    "finish_reason": last.finish_reason, "generation_tps": round(last.generation_tps, 1),
                    "peak_memory_gb": round(last.peak_memory, 3)}
        r = score(response, ex)
        rows.append({
            "dev_index": i, "db_id": ex["db_id"], "question": ex["question"], "gold": ex["query"],
            "response": response, "sql": r["sql"], "formatted": r["formatted"],
            "status": "api_error" if "api_error" in info else error_type(r), "error": r["error"],
            "correct": r["status"] == "correct", "gold_empty": gold_result(ex["db_id"], ex["query"])[0] == [],
            "latency_s": round(latency, 4), **info,
        })
        bar.set_postfix(acc=f"{sum(x['correct'] for x in rows) / len(rows):.3f}")

    n = len(rows)
    lat = sorted(x["latency_s"] for x in rows)
    non_empty = [x for x in rows if not x["gold_empty"]]
    summary = {
        "model": args.model, "adapter": args.adapter, "tag": args.tag,
        "split": args.split, "n": n, "subset_seed": args.seed if args.n else None,
        "decoding": f"API default sampling, effort {args.effort}" if args.api_model else "greedy",
        "max_tokens": 16000 if args.api_model else args.max_tokens,
        "execution_accuracy": round(sum(x["correct"] for x in rows) / n, 4),
        "execution_accuracy_excl_empty_gold": round(
            sum(x["correct"] for x in non_empty) / max(len(non_empty), 1), 4),
        "n_excl_empty_gold": len(non_empty),
        "sql_block_rate": round(sum(x["formatted"] for x in rows) / n, 4),
        "status_counts": dict(Counter(x["status"] for x in rows).most_common()),
        "latency_s": {"mean": round(statistics.mean(lat), 3), "median": round(statistics.median(lat), 3),
                      "p95": lat[int(0.95 * (n - 1))], "max": lat[-1]},
        "mean_prompt_tokens": round(statistics.mean(x.get("prompt_tokens", 0) for x in rows), 1),
        "mean_completion_tokens": round(statistics.mean(x.get("completion_tokens", 0) for x in rows), 1),
        "truncated_at_max_tokens": sum(x.get("finish_reason") == "length" for x in rows),
        "wall_time_s": round(time.perf_counter() - t_start, 1),
        "date": time.strftime("%Y-%m-%d"),
    }
    if args.api_model:
        costs = [x["cost_usd"] for x in rows if "cost_usd" in x]
        summary.update({
            "api_errors": sum("api_error" in x for x in rows), "refusals": sum(x.get("refusal", False) for x in rows),
            "served_by": dict(Counter(x.get("served_by") for x in rows if x.get("served_by"))),
            "cost_usd_total": round(sum(costs), 4),
            "cost_usd_per_request": round(statistics.mean(costs), 6) if costs else None,
            "prices_usd_per_mtok": API_PRICES.get(args.api_model), "anthropic_sdk": version("anthropic"),
        })
    else:
        summary.update({"peak_memory_gb": max(x["peak_memory_gb"] for x in rows),
                        "hardware": f"{platform.machine()} {platform.mac_ver()[0]}", "mlx_lm": version("mlx-lm")})
    name = (f"{args.tag}_{args.model.rstrip('/').split('/')[-1]}" + (f"_n{args.n}" if args.n else "")
            + ("_val" if args.split == "val" else ""))
    out = ROOT / "results" / f"{name}.json"
    out.write_text(json.dumps({"summary": summary, "examples": rows}, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2))
    print("written:", out)


if __name__ == "__main__":
    main()

"""Reliability layer for a text-to-SQL model: retry on SQL errors, self-consistency voting
and abstention ("I am not sure") instead of a confident wrong answer.

Why: in the final test, 22.8 % of the RL model's answers ran fine and returned a plausible
but wrong result. A business user cannot see that. A usable system must know when it does
not know. GRPO already samples several answers per question during training; at inference
we do the same and let the database results vote:

  1. greedy answer; if SQLite reports an error or the result is empty, show the model and let
     it fix the query (up to --retries times)
  2. --samples extra answers with temperature 0.8
  3. execute all, group them by their result rows, the largest group wins
  4. confidence = share of all answers in the winning group; below a threshold: abstain

From one run we report every strategy (greedy, +retry, +vote, +abstention), the
coverage/accuracy trade-off for all thresholds, and pass@k as an upper bound.

    python reliable.py --split versicherung --adapter checkpoints/grpo_local_binary_lr1e-6_g8_kl --tag rl
    python reliable.py --split versicherung --catalog --tag rl_catalog --adapter ...
    python reliable.py --split val --n 100 --tag base
    python reliable.py --backend hf --split versicherung --tag base_hf   # Linux / CUDA / CPU, no MLX

Writes results/reliable_<tag>_<split>.json (summary + every candidate of every question).
"""
import argparse
import hashlib
import json
import statistics
import time
from collections import Counter

from sql_env import (ROOT, SYSTEM_PROMPT, _normalize, build_prompt, execute_sql_full, extract_sql,
                     load_split, load_val, score)

SAMPLE_TEMPERATURE = 0.8   # same as in GRPO training
RETRY_PROMPT = ("The query failed with this SQLite error:\n{error}\n"
                "Write a corrected query. Output only the query inside a ```sql code block.")
EMPTY_PROMPT = ("The query ran, but the result is empty (or only 0 / NULL). Check the table names, columns "
                "and code values against the schema. Write a corrected query. "
                "Output only the query inside a ```sql code block.")


# ---------------------------------------------------------------- model backends

class MLXGenerator:
    """Apple Silicon (the project's Mac setup). Samples are drawn one after another."""

    def __init__(self, model, adapter=None, max_tokens=256):
        from mlx_lm import load
        self.model, self.tok = load(model, adapter_path=adapter)
        self.max_tokens = max_tokens

    def __call__(self, messages, n=1, temperature=0.0):
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
        prompt = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        sampler = make_sampler(temp=temperature, top_p=0.95 if temperature else 0.0)
        return [generate(self.model, self.tok, prompt, max_tokens=self.max_tokens, sampler=sampler)
                for _ in range(n)]


class HFGenerator:
    """transformers backend (CUDA or CPU). `model` may be a merged checkpoint from train_cloud.py;
    `adapter` a PEFT LoRA directory. All samples of one question in a single batch."""

    def __init__(self, model, adapter=None, max_tokens=256):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
            model, dtype=torch.bfloat16 if dev == "cuda" else torch.float32).to(dev).eval()
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter).merge_and_unload()
        self.max_tokens = max_tokens

    def __call__(self, messages, n=1, temperature=0.0):
        text = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        enc = self.tok(text, return_tensors="pt").to(self.model.device)
        kw = dict(do_sample=True, temperature=temperature, top_p=0.95, top_k=20, num_return_sequences=n) \
            if temperature else dict(do_sample=False, num_return_sequences=1)
        outs = []
        with self.torch.no_grad():
            for _ in range(1 if temperature else n):
                ids = self.model.generate(**enc, max_new_tokens=self.max_tokens, pad_token_id=self.tok.eos_token_id,
                                          **kw)
                outs += [self.tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True) for o in ids]
        return outs


def make_generator(backend, model, adapter=None, max_tokens=256, seed=0):
    if backend == "mlx":
        import mlx.core as mx
        mx.random.seed(seed)
        return MLXGenerator(model, adapter, max_tokens)
    import torch
    torch.manual_seed(seed)
    return HFGenerator(model, adapter, max_tokens)


# ---------------------------------------------------------------- the reliability layer

def result_key(rows):
    """Order-insensitive fingerprint of a result, so equivalent queries land in one group."""
    norm = sorted(map(repr, _normalize(rows)))
    return hashlib.sha1("\n".join(norm).encode()).hexdigest()[:12]


def is_degenerate(rows):
    """Empty result, only NULLs, or a single 0. Wrong filters (STAT_CD = 'Aktiv' instead of 'AKT')
    produce exactly these, and many different wrong queries then agree on the same 'result'.
    So a degenerate result is no evidence and gets no vote."""
    vals = [v for r in rows for v in r]
    return not vals or all(v is None for v in vals) or vals == [0]


def run_candidate(db_id, completion, max_rows=200):
    sql, _fmt = extract_sql(completion)
    if not sql:
        return {"sql": "", "error": "no SQL in answer", "key": None, "degenerate": False}
    rows, cols, err = execute_sql_full(db_id, sql)
    if err:
        return {"sql": sql, "error": err, "key": None, "degenerate": False}
    deg = is_degenerate(rows)
    return {"sql": sql, "error": None, "key": None if deg else result_key(rows), "degenerate": deg,
            "columns": cols, "rows": [list(r) for r in rows[:max_rows]], "n_rows": len(rows)}


def answer(gen, messages, db_id, samples=8, retries=2):
    """Full pipeline for one question. Returns the chosen answer, its confidence and all candidates."""
    t0 = time.perf_counter()
    first = gen(messages, 1, 0.0)[0]
    greedy = run_candidate(db_id, first)
    greedy["stage"] = "greedy"
    t_greedy = time.perf_counter() - t0
    chain, final, conv = [greedy], greedy, list(messages)
    completion = first
    for _ in range(retries):  # show the model its own SQLite error or empty result, let it fix the query
        if final["error"] == "no SQL in answer" or not (final["error"] or final["degenerate"]):
            break
        feedback = RETRY_PROMPT.format(error=final["error"]) if final["error"] else EMPTY_PROMPT
        conv = conv + [{"role": "assistant", "content": completion}, {"role": "user", "content": feedback}]
        completion = gen(conv, 1, 0.0)[0]
        final = run_candidate(db_id, completion)
        final["stage"] = "retry"
        chain.append(final)
    t_retry = time.perf_counter() - t0
    sampled = []
    for c in (gen(messages, samples, SAMPLE_TEMPERATURE) if samples else []):
        r = run_candidate(db_id, c)
        r["stage"] = "sample"
        sampled.append(r)
    t_all = time.perf_counter() - t0

    voters = [final] + sampled
    groups = Counter(c["key"] for c in voters if c["key"])
    if groups:
        top = max(groups.values())
        # tie: prefer the group of the (retried) greedy answer, then the first one seen
        winner = final["key"] if final["key"] and groups[final["key"]] == top else \
            next(c["key"] for c in voters if c["key"] and groups[c["key"]] == top)
        chosen = next(c for c in voters if c["key"] == winner)
        confidence = groups[winner] / len(voters)
    else:  # nothing usable: errors and empty results only -> confidence 0, the system abstains
        chosen, confidence = final, 0.0
    alternatives = [{"sql": next(c["sql"] for c in voters if c["key"] == k), "votes": v}
                    for k, v in groups.most_common() if k != chosen.get("key")]
    return {"chosen": chosen, "confidence": round(confidence, 4), "greedy": greedy, "final": final,
            "chain": chain, "samples": sampled, "alternatives": alternatives, "n_voters": len(voters),
            "latency_s": {"greedy": round(t_greedy, 3), "with_retry": round(t_retry, 3), "with_vote": round(t_all, 3)}}


# ---------------------------------------------------------------- evaluation

def is_correct(example, cand):
    """Answerable question: execution match with the gold query. Unanswerable: never correct
    to answer (abstaining is handled by the caller)."""
    if example["query"] is None or not cand.get("sql"):
        return False
    return score(f"```sql\n{cand['sql']}\n```", example)["status"] == "correct"


def load_examples(split, catalog=False):
    """Returns (examples, messages per example)."""
    if split == "versicherung":
        from insurance_db import load_questions, prompt_schema
        schema = prompt_schema(catalog)
        exs = load_questions()
        msgs = [[{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": f"Database schema:\n{schema}\n\nQuestion: {ex['question']}"}] for ex in exs]
        return exs, msgs
    exs = load_split("dev") if split == "dev" else load_val()
    return exs, [build_prompt(ex) for ex in exs]


def risk_coverage(records, total, n_voters):
    """For each minimum number of agreeing answers (k of n_voters): share of questions answered,
    accuracy on those, and how many unanswerable questions were correctly declined.
    Counted in votes, not in rounded confidences, so k/9 thresholds are exact."""
    table = []
    for k in range(1, n_voters + 1):
        answered = [r for r in records if r["votes"] >= k and r["vote_answer_runs"]]
        unans = [r for r in records if r["unanswerable"]]
        table.append({
            "min_votes": f"{k}/{n_voters}", "threshold": round(k / n_voters, 4),
            "coverage": round(len(answered) / total, 4),
            "accuracy_answered": round(sum(r["vote_correct"] for r in answered) / len(answered), 4) if answered else None,
            "wrong_answers": sum(not r["vote_correct"] for r in answered),
            "unanswerable_declined": f"{sum(r not in answered for r in unans)}/{len(unans)}" if unans else None,
        })
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="versicherung", choices=["versicherung", "val", "dev"])
    ap.add_argument("--backend", default="mlx", choices=["mlx", "hf"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--catalog", action="store_true", help="versicherung only: column descriptions + glossary in the prompt")
    ap.add_argument("--n", type=int, default=0, help="first N examples only (quick check)")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()

    exs, msgs = load_examples(args.split, args.catalog)
    if args.n:
        exs, msgs = exs[:args.n], msgs[:args.n]
    gen = make_generator(args.backend, args.model, args.adapter, args.max_tokens, args.seed)
    records, t_start = [], time.perf_counter()
    for i, (ex, m) in enumerate(zip(exs, msgs)):
        res = answer(gen, m, ex["db_id"], args.samples, args.retries)
        unans = ex["query"] is None
        chosen = res["chosen"]
        rec = {
            "id": ex.get("id", i), "question": ex["question"], "gold": ex["query"], "unanswerable": unans,
            "level": ex.get("level"), "lang": ex.get("lang"), "skills": ex.get("skills"),
            "greedy_sql": res["greedy"]["sql"], "greedy_error": res["greedy"]["error"],
            "greedy_correct": is_correct(ex, res["greedy"]),
            "retry_used": len(res["chain"]) > 1, "retry_sql": res["final"]["sql"],
            "retry_correct": is_correct(ex, res["final"]),
            "vote_sql": chosen["sql"], "vote_confidence": res["confidence"],
            "votes": round(res["confidence"] * res["n_voters"]),
            "vote_answer_runs": chosen["key"] is not None,
            "vote_correct": is_correct(ex, chosen),
            "pass_at_k": any(is_correct(ex, c) for c in [res["final"]] + res["samples"]),
            "candidates": [{"stage": c["stage"], "sql": c["sql"], "error": c["error"], "key": c["key"],
                            "degenerate": c["degenerate"]}
                           for c in res["chain"] + res["samples"]],
            "latency_s": res["latency_s"],
        }
        records.append(rec)
        done = len(records)
        print(f"[{done}/{len(exs)}] greedy {sum(r['greedy_correct'] for r in records)/done:.2f} "
              f"retry {sum(r['retry_correct'] for r in records)/done:.2f} vote {sum(r['vote_correct'] for r in records)/done:.2f} "
              f"conf {rec['vote_confidence']:.2f} | {ex['question'][:60]}", flush=True)

    n = len(records)
    answerable = [r for r in records if not r["unanswerable"]]
    acc = lambda key, rs=answerable: round(sum(r[key] for r in rs) / max(len(rs), 1), 4)
    by = lambda field: {k: {"n": len(v), "greedy": acc("greedy_correct", v), "vote": acc("vote_correct", v)}
                        for k, v in _group(answerable, field).items()}
    summary = {
        "split": args.split, "catalog": args.catalog, "model": args.model, "adapter": args.adapter,
        "backend": args.backend, "samples": args.samples, "retries": args.retries, "n": n,
        "n_answerable": len(answerable), "n_unanswerable": n - len(answerable),
        "accuracy_on_answerable": {"greedy": acc("greedy_correct"), "greedy_retry": acc("retry_correct"),
                                   "vote": acc("vote_correct"), f"pass_at_{args.samples + 1}": acc("pass_at_k")},
        "retry_used": sum(r["retry_used"] for r in records),
        "risk_coverage": risk_coverage(records, n, args.samples + 1),
        "by_level": by("level") if args.split == "versicherung" else None,
        "by_lang": by("lang") if args.split == "versicherung" else None,
        "latency_s_mean": {k: round(statistics.mean(r["latency_s"][k] for r in records), 3)
                           for k in ("greedy", "with_retry", "with_vote")},
        "wall_time_s": round(time.perf_counter() - t_start, 1), "date": time.strftime("%Y-%m-%d"),
    }
    out = ROOT / "results" / f"reliable_{args.tag}_{args.split}{'_catalog' if args.catalog else ''}.json"
    out.write_text(json.dumps({"summary": summary, "examples": records}, indent=1, ensure_ascii=False, default=str))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("written:", out)


def _group(rows, field):
    out = {}
    for r in rows:
        out.setdefault(r[field], []).append(r)
    return out


if __name__ == "__main__":
    main()

"""Spider text-to-SQL environment: data, prompts, sandboxed SQL execution, reward.

Data source: the official Spider release (Yale LILY), which includes the SQLite
databases. The Hugging Face dataset `xlangai/spider` only has questions and SQL,
no databases, so we don't use it.

    python sql_env.py download   # fetch + unzip into data/spider_data
    python sql_env.py examples   # print 5 example prompts
    python sql_env.py check      # run all gold queries + reward-hacking probes
"""
import json
import random
import re
import sqlite3
import subprocess
import sys
import time
import zipfile
from collections import Counter
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
SPIDER_DIR = DATA_DIR / "spider_data"
# Google Drive file ID linked from https://yale-lily.github.io/spider
SPIDER_DRIVE_ID = "1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J"

# Train = train_spider + train_others. Dev is the fixed, held-out test set:
# never use it for training or tuning. Its 20 databases don't appear in train.
SPLITS = {
    "train": ["train_spider.json", "train_others.json"],
    "dev": ["dev.json"],
}

SYSTEM_PROMPT = (
    "You are a SQLite expert. Given a database schema and a question, "
    "write one SQLite query that answers the question. "
    "Output only the query inside a ```sql code block, with no explanation."
)


def download_spider():
    DATA_DIR.mkdir(exist_ok=True)
    zip_path = DATA_DIR / "spider_data.zip"
    if not zip_path.exists():
        url = (f"https://drive.usercontent.google.com/download?id={SPIDER_DRIVE_ID}"
               "&export=download&confirm=t")
        subprocess.run(["curl", "-sSL", "-o", str(zip_path), url], check=True)
    if not zipfile.is_zipfile(zip_path):
        zip_path.unlink()
        sys.exit("Download failed (not a zip). Check the link on https://yale-lily.github.io/spider")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(DATA_DIR, members=[m for m in z.namelist() if not m.startswith("__MACOSX")])
    print("Spider extracted to", SPIDER_DIR)


def load_split(split):
    """Return a list of dicts with keys: db_id, question, query."""
    examples = []
    for fname in SPLITS[split]:
        for x in json.loads((SPIDER_DIR / fname).read_text()):
            examples.append({"db_id": x["db_id"], "question": x["question"], "query": x["query"]})
    return examples


def db_path(db_id):
    return SPIDER_DIR / "database" / db_id / f"{db_id}.sqlite"


@lru_cache(maxsize=None)
def get_schema(db_id):
    """CREATE TABLE statements as stored in the SQLite file, comments stripped."""
    con = sqlite3.connect(f"file:{db_path(db_id)}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    con.close()
    statements = []
    for (sql,) in rows:
        sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.S)
        sql = re.sub(r"--[^\n]*", "", sql)
        lines = [ln.strip() for ln in sql.splitlines() if ln.strip()]
        statements.append("\n".join(lines) + ";")
    return "\n\n".join(statements)


def build_prompt(example):
    """Chat messages; apply the tokenizer's chat template to get the model input."""
    user = f"Database schema:\n{get_schema(example['db_id'])}\n\nQuestion: {example['question']}"
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user}]


# ---------------------------------------------------------------- SQL extraction

_SQL_BLOCK = re.compile(r"```sql[ \t]*\n?(.*?)```", re.S | re.I)
_ANY_BLOCK = re.compile(r"```[a-zA-Z]*[ \t]*\n?(.*?)```", re.S)
_RAW_SQL = re.compile(r"^[ \t]*((?:SELECT|WITH)\b.*)", re.S | re.I | re.M)


def extract_sql(text):
    """Return (sql, formatted). formatted=True only for a closed ```sql block.

    Takes the last ```sql block; falls back to any code block, then to raw text
    starting with SELECT/WITH on its own line (cut at the first ';').
    """
    for pattern, formatted in ((_SQL_BLOCK, True), (_ANY_BLOCK, False)):
        blocks = [b.strip() for b in pattern.findall(text) if b.strip()]
        if blocks:
            return blocks[-1], formatted
    m = _RAW_SQL.search(text)
    if m:
        sql = m.group(1).split("```")[0]
        return (sql.split(";")[0] + ";").strip() if ";" in sql else sql.strip(), False
    return "", False


# ---------------------------------------------------------------- execution

EXEC_TIMEOUT = 5.0  # seconds, per query
# Authorizer whitelist: plain reads only. Blocks writes, ATTACH, PRAGMA, DDL,
# on top of the read-only connection (defense in depth).
_ALLOWED = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ,
            sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}


def _authorizer(action, *_):
    return sqlite3.SQLITE_OK if action in _ALLOWED else sqlite3.SQLITE_DENY


def execute_sql(db_id, sql, timeout=EXEC_TIMEOUT):
    """Run one query read-only. Returns (rows, error); error is None, 'timeout' or a message."""
    con = sqlite3.connect(f"file:{db_path(db_id)}?mode=ro", uri=True)
    con.text_factory = lambda b: b.decode(errors="replace")  # some Spider DBs have bad UTF-8
    con.set_authorizer(_authorizer)
    deadline = time.perf_counter() + timeout
    # Called every 10k VM steps; a truthy return aborts the query (also during fetch).
    con.set_progress_handler(lambda: time.perf_counter() > deadline, 10_000)
    try:
        return con.execute(sql).fetchall(), None
    except Exception as e:  # never let a bad completion crash training/eval
        return None, "timeout" if time.perf_counter() > deadline else f"{type(e).__name__}: {e}"
    finally:
        con.close()


@lru_cache(maxsize=20_000)
def gold_result(db_id, query):
    """Gold results are re-used across many generations, so cache them."""
    return execute_sql(db_id, query, timeout=30.0)


def _normalize(rows):
    return [tuple(round(v, 6) if isinstance(v, float) else v for v in r) for r in rows]


def results_match(pred_rows, gold_rows, ordered):
    """Row order matters only if the gold query has ORDER BY; otherwise multiset compare.
    Column order always matters."""
    pred_rows, gold_rows = _normalize(pred_rows), _normalize(gold_rows)
    if ordered:
        return pred_rows == gold_rows
    return Counter(pred_rows) == Counter(gold_rows)


def has_order_by(query):
    return re.search(r"\border\s+by\b", query, re.I) is not None


# ---------------------------------------------------------------- reward

def score(completion, example, format_bonus=0.0, timeout=EXEC_TIMEOUT):
    """Grade one model completion. Returns a dict with reward and diagnostics.

    reward = 1.0 if the execution result matches the gold result,
             format_bonus if the answer is a ```sql block that executes but is wrong,
             0.0 otherwise (no SQL, error, timeout).
    status in: correct, wrong_result, exec_error, timeout, no_sql, gold_error
    """
    sql, formatted = extract_sql(completion)
    out = {"sql": sql, "formatted": formatted, "status": None, "error": None, "reward": 0.0}
    if not sql:
        out["status"] = "no_sql"
        return out
    gold_rows, gold_err = gold_result(example["db_id"], example["query"])
    if gold_err:
        out.update(status="gold_error", error=gold_err)
        return out
    rows, err = execute_sql(example["db_id"], sql, timeout)
    if err:
        out.update(status="timeout" if err == "timeout" else "exec_error", error=err)
    elif results_match(rows, gold_rows, has_order_by(example["query"])):
        out.update(status="correct", reward=1.0)
    else:
        out.update(status="wrong_result", reward=format_bonus if formatted else 0.0)
    return out


def reward(completion, example, format_bonus=0.0):
    return score(completion, example, format_bonus)["reward"]


# Reward engineering variants (cf. grid-world lecture: goal reward vs. -1 per timestep).
REWARD_VARIANTS = ("binary", "format", "length")
LENGTH_PENALTY_PER_CHAR = 0.0002  # 1,000 characters (~250 tokens) cost the maximum
LENGTH_PENALTY_MAX = 0.2


def shaped_reward(completion, example, variant="binary", result=None):
    """binary: 1/0. format: 0.1 for a wrong but executable ```sql block.
    length: binary minus a small per-character cost (max 0.2), so shorter correct
    answers win and runaway generations are discouraged.
    Pass `result` (from score) to avoid executing the query twice."""
    if variant not in REWARD_VARIANTS:
        raise ValueError(f"unknown reward variant {variant!r}, choose from {REWARD_VARIANTS}")
    result = result or score(completion, example)
    r = 1.0 if result["status"] == "correct" else 0.0
    if variant == "format" and result["status"] == "wrong_result" and result["formatted"]:
        r = 0.1
    elif variant == "length":
        r -= min(LENGTH_PENALTY_MAX, LENGTH_PENALTY_PER_CHAR * len(completion))
    return r


# ---------------------------------------------------------------- training log (MLX and TRL)

class RewardLog:
    """Grades one batch of completions and records statistics for curves and the trend test.

    Completions must be grouped by question: `group_size` consecutive answers per question
    (true for mlx-lm-lora and for TRL's RepeatSampler). Framework-independent, so the local
    MLX trainer and the TRL cloud trainer log identically."""

    def __init__(self, variant, group_size):
        if variant not in REWARD_VARIANTS:
            raise ValueError(f"unknown reward variant {variant!r}")
        self.variant, self.group_size, self.steps = variant, group_size, []

    def grade(self, completions, examples):
        import statistics
        results = [score(c, ex) for c, ex in zip(completions, examples)]
        rewards = [shaped_reward(c, ex, self.variant, r) for c, ex, r in zip(completions, examples, results)]
        g = self.group_size
        pick = random.randrange(len(completions))
        row = {
            "step": len(self.steps) + 1,
            "reward_mean": statistics.mean(rewards),
            "accuracy": statistics.mean(r["status"] == "correct" for r in results),
            # GRPO learns nothing from a group whose answers all got the same reward
            "no_signal_share": statistics.mean(len(set(rewards[i:i + g])) == 1 for i in range(0, len(rewards), g)),
            "mean_chars": statistics.mean(len(c) for c in completions),
            "sql_block_share": statistics.mean(r["formatted"] for r in results),  # format drift check
            "status": dict(Counter(r["status"] for r in results)),
            # per question: share of its answers that were correct (for the trend test)
            "per_question": [{"question": examples[i]["question"],
                              "accuracy": statistics.mean(r["status"] == "correct" for r in results[i:i + g])}
                             for i in range(0, len(results), g)],
            "sample": {"question": examples[pick]["question"], "completion": completions[pick],
                       "status": results[pick]["status"], "reward": rewards[pick]},
        }
        self.steps.append(row)
        return rewards, row


def within_question_trend(steps):
    """Accuracy trend per 100 steps (as a fraction), comparing each question only with itself.

    Regresses per-question accuracy on the step number with a fixed effect per question
    (questions seen at least twice), so batch difficulty cannot fake a trend."""
    import math
    import statistics
    visits = {}
    for s in steps:
        for q in s.get("per_question", []):
            visits.setdefault(q["question"], []).append((s["step"], q["accuracy"]))
    groups = [v for v in visits.values() if len(v) >= 2]
    if not groups:
        return None
    num = den = 0.0
    centred = []
    for v in groups:
        mt, ma = statistics.mean(t for t, _ in v), statistics.mean(a for _, a in v)
        centred.append([(t - mt, a - ma) for t, a in v])
        num += sum(dt * da for dt, da in centred[-1])
        den += sum(dt * dt for dt, _ in centred[-1])
    if den == 0:
        return None
    beta = num / den
    resid = [da - beta * dt for c in centred for dt, da in c]
    dof = max(len(resid) - len(groups) - 1, 1)
    se = math.sqrt(sum(r * r for r in resid) / dof / den)
    return {"per_100_steps": round(100 * beta, 4), "se": round(100 * se, 4),
            "t": round(beta / se, 2) if se else None, "questions": len(groups), "visits": len(resid)}


# ---------------------------------------------------------------- checks

def get_tables(db_id):
    con = sqlite3.connect(f"file:{db_path(db_id)}?mode=ro", uri=True)
    names = [r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
    con.close()
    return names


@lru_cache(maxsize=None)
def db_is_empty(db_id):
    con = sqlite3.connect(f"file:{db_path(db_id)}?mode=ro", uri=True)
    n = sum(con.execute(f'SELECT count(*) FROM "{t}"').fetchone()[0] for t in get_tables(db_id))
    con.close()
    return n == 0


def _rl_usable():
    """Train examples where an execution reward is informative. Drops examples whose
    DB has no rows at all (7 legacy DBs: geo, scholar, ...), whose gold result is empty
    (an empty-result query would score 1.0) or whose gold query fails."""
    keep = []
    for ex in load_split("train"):
        if db_is_empty(ex["db_id"]):
            continue
        rows, err = gold_result(ex["db_id"], ex["query"])
        if not err and rows:
            keep.append(ex)
    return keep


VAL_DBS, VAL_PER_DB = 10, 25


@lru_cache(maxsize=None)
def val_db_ids():
    """10 validation databases carved out of train (random, seed 0).
    All tuning decisions use this split; dev stays untouched until the final comparison.
    Like dev, its databases never appear in training."""
    dbs = sorted({ex["db_id"] for ex in _rl_usable()})
    random.Random(0).shuffle(dbs)
    return frozenset(dbs[:VAL_DBS])


def load_rl_train():
    """Filtered train examples (see _rl_usable) minus the validation databases."""
    return [ex for ex in _rl_usable() if ex["db_id"] not in val_db_ids()]


def load_val():
    """At most 25 questions per validation database (seed 0), so no single DB dominates."""
    by_db = {}
    for ex in _rl_usable():
        if ex["db_id"] in val_db_ids():
            by_db.setdefault(ex["db_id"], []).append(ex)
    out = []
    for db in sorted(by_db):
        exs = by_db[db]
        out += random.Random(0).sample(exs, min(VAL_PER_DB, len(exs)))
    return out


def env_check():
    """Run every gold query, then measure how often trivial 'hack' queries would score 1.0."""
    report = {}
    for split, loader in (("train", lambda: load_split("train")), ("rl_train", load_rl_train),
                          ("dev", lambda: load_split("dev"))):
        exs = loader()
        gold_errors, empty, slowest, gold_keys = [], [], (0.0, None), Counter()
        for ex in exs:
            t0 = time.perf_counter()
            rows, err = gold_result(ex["db_id"], ex["query"])
            dt = time.perf_counter() - t0
            slowest = max(slowest, (dt, ex["query"]), key=lambda x: x[0])
            if err:
                gold_errors.append({**ex, "error": err})
            elif not rows:
                empty.append(ex)
            else:
                gold_keys[repr(sorted(map(repr, _normalize(rows))))] += 1

        # Probes: constant queries, and 'oracle' table choice (best of all tables)
        # as an upper bound for SELECT count(*) / SELECT * tricks.
        probe_hits = Counter()
        by_db = {}
        for ex in exs:
            by_db.setdefault(ex["db_id"], []).append(ex)
        for db_id, db_exs in by_db.items():
            tables = get_tables(db_id)
            probes = {
                "SELECT 1 WHERE 0 (empty result)": ["SELECT 1 WHERE 0"],
                "SELECT 1": ["SELECT 1"],
                "SELECT count(*) FROM <best table>": [f'SELECT count(*) FROM "{t}"' for t in tables],
                "SELECT * FROM <best table>": [f'SELECT * FROM "{t}"' for t in tables],
            }
            probe_rows = {name: [execute_sql(db_id, q)[0] for q in qs] for name, qs in probes.items()}
            for ex in db_exs:
                gold_rows, err = gold_result(ex["db_id"], ex["query"])
                if err:
                    continue
                ordered = has_order_by(ex["query"])
                for name, results in probe_rows.items():
                    if any(r is not None and results_match(r, gold_rows, ordered) for r in results):
                        probe_hits[name] += 1

        report[split] = {
            "n": len(exs),
            "gold_errors": gold_errors,
            "gold_empty_result": len(empty),
            "gold_empty_examples": empty[:5],
            "slowest_gold_s": round(slowest[0], 3),
            "slowest_gold_query": slowest[1],
            "probe_hit_rate": {k: round(v / len(exs), 4) for k, v in probe_hits.items()},
            "most_common_gold_results": [
                {"result": k[:120], "count": c, "share": round(c / len(exs), 4)}
                for k, c in gold_keys.most_common(5)],
        }
        gold_result.cache_clear()
    return report


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "examples"
    if cmd == "download":
        download_spider()
    elif cmd == "check":
        rep = env_check()
        out = ROOT / "results" / "phase2_env_check.json"
        out.write_text(json.dumps(rep, indent=2, default=str))
        for split, r in rep.items():
            print(f"{split}: n={r['n']} gold_errors={len(r['gold_errors'])} "
                  f"gold_empty={r['gold_empty_result']} slowest_gold={r['slowest_gold_s']}s")
            for k, v in r["probe_hit_rate"].items():
                print(f"   {v:7.2%}  {k}")
            for m in r["most_common_gold_results"]:
                print(f"   {m['share']:7.2%}  gold result {m['result']}")
        print("written:", out)
    elif cmd == "examples":
        train, dev = load_split("train"), load_split("dev")
        print(f"train: {len(train)} examples, dev: {len(dev)} examples\n")
        for ex in [train[0], train[3000], train[7500], dev[0], dev[500]]:
            print("=" * 80)
            print(f"db_id: {ex['db_id']}   gold: {ex['query']}")
            print("-" * 80)
            for m in build_prompt(ex):
                print(f"[{m['role']}]\n{m['content']}\n")

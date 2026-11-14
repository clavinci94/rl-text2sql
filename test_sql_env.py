"""Unit tests for the reward function. Run: .venv/bin/python test_sql_env.py (or pytest)."""
import hashlib
import time

from sql_env import db_path, extract_sql, load_split, score

SINGERS = {"db_id": "concert_singer", "question": "How many singers do we have?",
           "query": "SELECT count(*) FROM singer"}
ORDERED = {"db_id": "concert_singer", "question": "Names of singers ordered by age descending.",
           "query": "SELECT name FROM singer ORDER BY age DESC"}
UNORDERED = {"db_id": "concert_singer", "question": "What are the names of all singers?",
             "query": "SELECT name FROM singer"}


def block(sql):
    return f"```sql\n{sql}\n```"


def test_gold_sql_gets_full_reward_on_all_dev_examples():
    dev = load_split("dev")
    bad = [ex for ex in dev if score(block(ex["query"]), ex)["reward"] != 1.0]
    assert not bad, f"{len(bad)} dev gold queries did not score 1.0, e.g. {bad[:2]}"


def test_gold_sql_sample_of_train():
    train = load_split("train")[::20]  # every 20th example, 433 queries
    bad = [ex for ex in train if score(block(ex["query"]), ex)["reward"] != 1.0]
    assert not bad, f"{len(bad)} train gold queries did not score 1.0, e.g. {bad[:2]}"


def test_equivalent_sql_and_surrounding_text():
    for completion in [
        block("select COUNT(*) from singer;"),
        block("SELECT count(Singer_ID) FROM singer"),
        "Here is the query:\n" + block("SELECT count(*) FROM singer") + "\nThis counts rows.",
    ]:
        assert score(completion, SINGERS)["reward"] == 1.0, completion


def test_wrong_sql_gets_zero():
    cases = {
        block("SELECT count(*) FROM stadium"): "wrong_result",      # wrong table
        block("SELECT count(*) FROM singer WHERE Age > 30"): "wrong_result",
        block("SELECT 1 WHERE 0"): "wrong_result",                   # empty result
        block("SELECT * FROM singer"): "wrong_result",               # SELECT * trick
        block("SELEC count(*) FROM singer"): "exec_error",           # syntax error
        block("SELECT count(*) FROM singers"): "exec_error",         # no such table
        block("SELECT 1; SELECT 2"): "exec_error",                   # multiple statements
        "I don't know.": "no_sql",
        "": "no_sql",
    }
    for completion, status in cases.items():
        r = score(completion, SINGERS)
        assert r["reward"] == 0.0 and r["status"] == status, (completion, r)


def test_write_attempts_are_blocked_and_db_unchanged():
    path = db_path("concert_singer")
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    for sql in ["DROP TABLE singer", "DELETE FROM singer", "UPDATE singer SET Age = 0",
                "INSERT INTO singer (Singer_ID) VALUES (999)",
                "CREATE TABLE x (a int)", "PRAGMA journal_mode=WAL",
                "ATTACH DATABASE 'file:/tmp/evil.db?mode=rwc' AS evil"]:
        r = score(block(sql), SINGERS)
        assert r["reward"] == 0.0 and r["status"] == "exec_error", (sql, r)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert not path.with_name(path.name + "-journal").exists()


def test_timeout():
    endless = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT count(*) FROM c"
    t0 = time.perf_counter()
    r = score(block(endless), SINGERS, timeout=1.0)
    elapsed = time.perf_counter() - t0
    assert r["status"] == "timeout" and r["reward"] == 0.0, r
    assert elapsed < 2.0, elapsed


def test_row_order_only_matters_with_order_by():
    assert score(block("SELECT name FROM singer ORDER BY age DESC"), ORDERED)["reward"] == 1.0
    assert score(block("SELECT name FROM singer ORDER BY age ASC"), ORDERED)["reward"] == 0.0
    assert score(block("SELECT name FROM singer ORDER BY name"), UNORDERED)["reward"] == 1.0


def test_format_bonus():
    wrong = block("SELECT count(*) FROM stadium")
    assert score(wrong, SINGERS, format_bonus=0.1)["reward"] == 0.1
    assert score("SELECT count(*) FROM stadium", SINGERS, format_bonus=0.1)["reward"] == 0.0  # no block
    assert score(block("SELEC x"), SINGERS, format_bonus=0.1)["reward"] == 0.0             # error
    assert score(block("SELECT count(*) FROM singer"), SINGERS, format_bonus=0.1)["reward"] == 1.0


def test_reward_variants():
    from sql_env import shaped_reward
    right, wrong = block("SELECT count(*) FROM singer"), block("SELECT count(*) FROM stadium")
    assert shaped_reward(right, SINGERS, "binary") == 1.0
    assert shaped_reward(wrong, SINGERS, "binary") == 0.0
    assert shaped_reward(wrong, SINGERS, "format") == 0.1
    short = shaped_reward(right, SINGERS, "length")
    longer = shaped_reward("Sure! Here is the query:\n" + right + "\nIt counts singers.", SINGERS, "length")
    assert 0.98 < short < 1.0 and longer < short                     # shorter correct answer wins
    assert shaped_reward("x" * 5000, SINGERS, "length") == -0.2        # penalty is capped
    # the loophole that collapsed training: under "length" an empty answer beats a wrong one
    assert shaped_reward("", SINGERS, "length") > shaped_reward(wrong, SINGERS, "length")
    # length_correct closes it: giving up = wrong = 0, shorter correct answers still win
    assert shaped_reward("", SINGERS, "length_correct") == shaped_reward(wrong, SINGERS, "length_correct") == 0.0
    long_right = "Sure! Here is the query:\n" + right + "\nIt counts singers."
    assert 1.0 > shaped_reward(right, SINGERS, "length_correct") > shaped_reward(long_right, SINGERS, "length_correct") > 0.9
    try:
        shaped_reward(right, SINGERS, "nope")
        assert False, "unknown variant must raise"
    except ValueError:
        pass


def test_rl_train_has_no_empty_or_broken_gold():
    from sql_env import db_is_empty, gold_result, load_rl_train
    from sql_env import load_val
    rl, val = load_rl_train(), load_val()
    assert 0 < len(rl) < len(load_split("train")) and len(val) >= 200
    rl_dbs, val_dbs = {ex["db_id"] for ex in rl}, {ex["db_id"] for ex in val}
    dev_dbs = {ex["db_id"] for ex in load_split("dev")}
    assert not rl_dbs & dev_dbs and not rl_dbs & val_dbs and not val_dbs & dev_dbs
    for ex in rl[::10]:
        rows, err = gold_result(ex["db_id"], ex["query"])
        assert err is None and rows and not db_is_empty(ex["db_id"]), ex


def test_extract_sql():
    assert extract_sql(block("SELECT 1")) == ("SELECT 1", True)
    assert extract_sql("```\nSELECT 1\n```") == ("SELECT 1", False)
    assert extract_sql("```sql\nSELECT 1") == ("SELECT 1", False)          # truncated block
    assert extract_sql("SELECT a FROM t; -- explanation") == ("SELECT a FROM t;", False)
    assert extract_sql(block("SELECT 1") + "\n" + block("SELECT 2")) == ("SELECT 2", True)
    assert extract_sql("Queries go with care.") == ("", False)


def test_insurance_case_study_gold_queries():
    from insurance_db import DB_FILE, build, check
    if not DB_FILE.exists():
        build()
    assert check(), "every answerable gold query must run and return a non-empty result"


def test_reliability_layer_votes_retries_and_abstains():
    """Stub model, real database: voting by result, retry after an SQLite error, no votes for
    empty results, abstention when nothing usable comes back."""
    from insurance_db import DB_FILE, build
    from reliable import answer
    if not DB_FILE.exists():
        build()
    ok = block("SELECT count(*) FROM T_SCHADEN WHERE STAT_CD IN ('OFF', 'PRF')")
    same = block("SELECT COUNT(SCH_ID) FROM T_SCHADEN WHERE STAT_CD = 'OFF' OR STAT_CD = 'PRF'")
    wrong_empty = block("SELECT count(*) FROM T_SCHADEN WHERE STAT_CD = 'offen'")   # -> 0, degenerate
    broken = block("SELECT count(*) FROM T_SCHADEN WHERE STATUS = 'OFF'")             # no such column

    def stub(script):
        calls = iter(script)
        return lambda messages, n=1, temperature=0.0: [next(calls) for _ in range(n)]

    msgs = [{"role": "system", "content": ""}, {"role": "user", "content": "offene Schäden?"}]
    # greedy fails -> retry fixes it; 4 samples: 2 equivalent, 2 empty (no votes)
    r = answer(stub([broken, ok, same, wrong_empty, same, wrong_empty]), msgs, "versicherung", samples=4, retries=2)
    assert r["greedy"]["error"] and not r["final"]["error"] and len(r["chain"]) == 2
    assert r["chosen"]["rows"] == [[211]] or r["chosen"]["key"] == r["final"]["key"]
    assert r["confidence"] == 3 / 5                                   # final + 2 equivalent samples
    # many wrong queries agreeing on "0" must not create confidence
    r = answer(stub([wrong_empty, wrong_empty, wrong_empty, wrong_empty, wrong_empty]), msgs, "versicherung",
               samples=4, retries=0)
    assert r["confidence"] == 0.0 and r["chosen"]["degenerate"]


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            t0 = time.perf_counter()
            try:
                fn()
                print(f"PASS {name} ({time.perf_counter() - t0:.1f}s)")
            except AssertionError as e:
                failed += 1
                print(f"FAIL {name}: {str(e)[:500]}")
    raise SystemExit(1 if failed else 0)

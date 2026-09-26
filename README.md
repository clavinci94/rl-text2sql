# rl-text2sql

RL fine-tuning (GRPO, verifiable execution reward) of a small LLM
(Qwen2.5-Coder-0.5B-Instruct, later 1.5B) for text-to-SQL on Spider.
Goal: compare base model vs. RL model on a held-out test set
(execution accuracy, latency, cost per request).

## Setup

```bash
/opt/homebrew/bin/python3.12 -m venv .venv   # Python 3.12; 3.14 is too new for some wheels
.venv/bin/pip install -r requirements.txt
```

## Data (Spider)

The Hugging Face dataset `xlangai/spider` contains only questions and gold SQL,
**not the SQLite databases**, so it cannot be used for execution-based rewards.
We use the official Spider release from https://yale-lily.github.io/spider
(Google Drive file `1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J`, ~206 MB):

```bash
.venv/bin/python sql_env.py download   # -> data/spider_data/ (gitignored)
.venv/bin/python sql_env.py examples   # print 5 example prompts
```

Downloaded zip SHA-256 (2026-09-24):
`00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b`

| Split | Files | Examples | Databases |
|---|---|---|---|
| train | `train_spider.json` + `train_others.json` | 8,659 | 146 |
| dev (held-out test set) | `dev.json` | 1,034 | 20 |

Dev and train share no databases. **Dev is never used for training or tuning.**
The zip also contains the official Spider test split (2,147 examples, 206 DBs),
which we don't use.

## Prompt format

System prompt asks for a single SQLite query inside a ` ```sql ` block.
User message = `CREATE TABLE` statements read from the SQLite file (comments
stripped) + the question. Prompt length with the Qwen chat template:
median 419 tokens (train) / 302 (dev), max 1,838 / 928.

## Reward (`sql_env.score`)

- SQL = last ` ```sql ` block (fallbacks: any code block, raw `SELECT`/`WITH`).
- Executed read-only (`mode=ro`) plus an authorizer that only allows SELECT/READ/FUNCTION,
  so writes, `ATTACH` and `PRAGMA` are rejected. 5 s timeout via progress handler.
- Match = same rows as the gold result (multiset; ordered only if gold has `ORDER BY`;
  column order matters; floats rounded to 6 decimals).
- Reward 1.0 if correct, otherwise 0.0; optional `format_bonus` (e.g. 0.1) for a
  ` ```sql ` block that executes but is wrong.

```bash
.venv/bin/python test_sql_env.py   # unit tests
.venv/bin/python sql_env.py check  # all gold queries + reward-hacking probes -> results/phase2_env_check.json
```

**Validation split (`load_val`, 231 examples).** 10 train databases (seed 0, at most 25
questions each) are held out of training. All tuning decisions use this split
(`evaluate.py --split val`); dev is reserved for the final comparison.

**RL training set (`load_rl_train`, 6,076 examples after removing the validation DBs).** 7 train DBs (geo, scholar,
academic, restaurants, imdb, music_2, yelp) contain no rows, and 1,616 train gold
queries return an empty result, so a trivial `SELECT 1 WHERE 0` would score on
18.7% of raw train. These examples plus 3 broken gold queries are dropped for RL.
Dev stays unchanged. Its 49 empty-gold examples (4.7%) remain a known
false-positive source for execution accuracy.

## Evaluation (`evaluate.py`)

Greedy decoding, max 256 new tokens, one request at a time (latency = prefill + generation).

```bash
.venv/bin/python evaluate.py --n 200   # fixed random dev subset (seed 0)
.venv/bin/python evaluate.py           # full dev -> results/baseline_<model>.json
```

| Model | Dev n | Exec. accuracy | excl. empty gold | Mean latency | Peak memory |
|---|---|---|---|---|---|
| Qwen2.5-Coder-0.5B-Instruct (base) | 1,034 | 45.3% | 45.0% | 0.60 s | 1.5 GB |

Apple M4, 24 GB, MLX 0.32.2 / mlx-lm 0.31.3, bf16 weights.

### Large API model baseline (Claude)

`--api-model` sends the same system prompt and user message to a Claude model through the
official Anthropic SDK (`anthropic` 1.8.0) and records tokens, latency and cost per request.
Default `claude-opus-5`, effort `high` (the API default); current Claude models do not accept
`temperature`, so sampling stays at the API default. Server-side refusal fallbacks are enabled
(`fallbacks="default"`); `served_by` in the results shows which model answered.
Prices are hard-coded in `evaluate.API_PRICES` (Anthropic first-party rates; check the live
pricing page before quoting).

```bash
.venv/bin/python evaluate.py --api-model claude-opus-5 --split val --dry-run   # prints the request, no call
export ANTHROPIC_API_KEY=...                                                   # or: ant auth login
.venv/bin/python evaluate.py --api-model claude-opus-5 --split val --tag api   # 231 requests, costs money
.venv/bin/python evaluate.py --api-model claude-opus-5 --tag api               # dev: final comparison only
```

For a cheaper large-scale run without latency measurement, the Message Batches API costs half.
Prompt caching is not used: on Claude Opus 5 only prefixes of at least 512 tokens cache, which
only the longer schemas reach, and it would need the schema in its own content block. Since each
dev database is asked about ~50 times, caching the long schemas could lower cost in production.

## Local GRPO smoke test (`train_local.py`)

mlx-lm has no GRPO; we use the `train_grpo` Python API of the community package
[mlx-lm-lora](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora) (pinned 3.1.3) with our own
reward from `sql_env.py`. LoRA on all linear layers, no reference model by default (`--beta 0`).

```bash
.venv/bin/python train_local.py                    # 200 train questions x 8 answers, 300 steps, ~70 min
.venv/bin/python train_local.py --reward format    # reward variants: binary | format | length | length_correct
.venv/bin/python evaluate.py --split val --adapter checkpoints/grpo_local_binary --tag grpo_local_binary
```

Defaults: LoRA rank 8 / scale 20 on all linear layers, lr 1e-6, KL beta 0.04, temperature 0.8,
max 256 new tokens. Results (validation split, 231 questions on 10 unseen databases):

| Run | Setting | Trend on train questions per 100 steps | Validation |
|---|---|---|---|
| base model | – | – | 49.4% |
| 1 | library loss, lr 1e-5, 4 answers | -5.7 points (t = -3.6) | 24.2%\* |
| 3 | fixed loss, lr 1e-5, 4 answers | +1.5 points (t = 1.1) | 37.2%\* |
| 4 | fixed loss, lr 1e-6, KL 0.04, 8 answers | +5.8 points (t = 8.8) | **56.7%** |

\* trained before the validation split existed; 23 / 29 of their 200 training questions come from
validation DBs, so these values are, if anything, too optimistic. Run 2 ran out of memory.
Run 4 vs. base on validation: 24 questions only run 4 solves, 7 only the base model (McNemar z = 3.05).

Outputs: `checkpoints/<run>/` (LoRA adapter), `results/<run>.json` (per-step log incl. one sample
answer per step for spot checks), `results/<run>_reward.png`.

Reward variants (`sql_env.shaped_reward`, cf. the grid-world reward-engineering lecture), each
trained with the run-4 setting on the same 200 questions in the same order:

| Variant | Reward | Validation | vs. base (McNemar z) | Tokens | ` ```sql ` block |
|---|---|---|---|---|---|
| base model | – | 49.4% | – | 32.8 | 100% |
| `binary` | 1 if correct, else 0 | 56.7% | +3.05 | 29.9 | 100% |
| `format` | + 0.1 for a wrong but executable ` ```sql ` block | 56.3% | +2.83 | 29.5 | 100% |
| `length_correct` | correct: 1 - 0.0002 per char (max 0.2); else 0 | 55.8% | +2.69 | 20.5 | 0% |
| `length` | binary - 0.0002 per char (max 0.2) on every answer | **0.0%** | collapsed | 1.0 | 0% |

The three working variants do not differ significantly from each other (|z| < 0.5).
`length` is reward-hacked: an empty answer (0) beats every wrong answer (< 0), group-std
normalization turns those tiny differences into full advantages, and from step 130 the model
answered everything with an empty string. `length_correct` closes that loophole but saves
characters by dropping the ` ```sql ` fences (still scored correct via the raw-SQL fallback).
`format` lowers the share of groups without learning signal from 68% to 45% at the same end result.
Figure: `results/reward_variants_comparison.png`. The collapsed run crashed at step ~210 on a
logging bug that is fixed now; its 10-step curve and step-200 evaluation are in
`results/grpo_local_length_lr1e-6_g8_kl_collapsed.json`.

Known issues in mlx-lm-lora 3.1.3 and how `train_local.py` handles them:

- **Bug: the built-in GRPO loss ignores the prompt.** `grpo_loss` computes the completion
  log-probs from the completion tokens alone, without schema and question, and never scores
  the first completion token. It reinforces SQL text regardless of the question; our first
  run degraded (accuracy trend -5.7 points per 100 steps, t = -3.6). `make_grpo_loss` passes
  a prompt-conditioned loss via `train_grpo(loss_fn=...)`. It matches a plain forward pass up
  to bf16 rounding and also trains the end-of-answer token.
- The reward function is called twice per step with identical inputs; `SQLReward` caches the
  second call.
- Each step draws a random batch with replacement (a new shuffled iterator per step), so
  there are no clean epochs.
- Scoring prompt + answer with gradients costs memory: prompts are capped at 1,024 tokens
  (dev max is 928) and gradient checkpointing is on by default.

## Cloud GRPO training (`train_cloud.py`)

TRL's `GRPOTrainer` with the same reward, data split and defaults as local run 4
(lr 1e-6 constant, KL beta 0.04, 8 answers per question, temperature 0.8, top-p 0.95, top-k 20,
LoRA rank 8 on all linear layers with `lora_alpha = 20 * rank`, TRL loss `dapo` = mean over all
answer tokens like the local loss). Both setups have exactly 4,399,104 trainable LoRA parameters.
Differences to the local run: 8 questions per optimizer step instead of 2, and one epoch
over all ~6,000 training questions instead of 200 questions.

Checked on the Mac without training (`--dry-run` builds data, reward, config and the full
trainer, grades 16 gold answers: all 1.00). Not yet run on a GPU.

### Run on a rented GPU

Any single NVIDIA GPU with at least 24 GB (e.g. RTX 4090, L4, A10G, L40S, A100, H100) is enough
for the 0.5B model; for 1.5B on 24 GB add `--grad-accum 2`.

```bash
# on the GPU machine (Linux + CUDA), with this repo copied or cloned
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-cloud.txt          # add wandb / vllm if needed
python sql_env.py download                     # Spider + SQLite databases (~206 MB)
python test_sql_env.py                         # reward sanity check, ~10 s
python train_cloud.py --dry-run                # everything except training
python train_cloud.py --max-steps 20           # short run: measure time per step first
tmux new -s grpo 'python train_cloud.py 2>&1 | tee train.log'   # full run: one epoch, 0.5B
# 1.5B, W&B logging, fast generation:
#   wandb login && python train_cloud.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct --lora-rank 16 --wandb --use-vllm
```

CLI (see `--help`): `--model`, `--num-generations`, `--lr`, `--max-steps`, `--lora-rank`,
`--beta`, `--reward {binary,format,length}`, `--prompts-per-step`, `--grad-accum`, `--wandb`, `--use-vllm`.

### Bring results back and evaluate on the Mac

```bash
rsync -av gpu:rl-text2sql/checkpoints/cloud_Qwen2.5-Coder-0.5B-Instruct_binary/{merged,steps.json,log_history.json} \
      checkpoints/cloud_Qwen2.5-Coder-0.5B-Instruct_binary/
.venv/bin/python evaluate.py --model checkpoints/cloud_Qwen2.5-Coder-0.5B-Instruct_binary/merged --split val --tag cloud
.venv/bin/python train_local.py --plot checkpoints/cloud_Qwen2.5-Coder-0.5B-Instruct_binary/steps.json
```

`merged/` is a normal Hugging Face model (LoRA merged in), which mlx-lm loads directly.
Note the GPU's hourly price and the run time from `steps.json`; both are needed for the
cost comparison. Evaluate on `dev` only for the final comparison.

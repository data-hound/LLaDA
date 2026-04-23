#!/usr/bin/env python3
"""
eval_sudoku_ar.py  —  Evaluate standard autoregressive HuggingFace models
                      on the GridCorpus Sudoku benchmark.

Designed to be a drop-in companion to eval_sudoku.py (LLaDA). Uses identical
dataset loading, prompting, metrics, and output format — the only difference
is generation: standard model.generate() (greedy by default) instead of
LLaDA's masked-diffusion sampler.

Tested with:
    meta-llama/Meta-Llama-3-8B-Instruct
    deepseek-ai/deepseek-math-7b-instruct

Metrics reported per difficulty
────────────────────────────────
  cell_acc   : fraction of originally-blank cells predicted correctly
  board_acc  : fraction of boards solved perfectly
  row_acc    : fraction of rows where every blank cell is correct
  col_acc    : fraction of columns where every blank cell is correct
  parse_fail : count / percent of responses that could not be parsed

Usage
─────
  # Llama-3, zero-shot, 20k per difficulty
  python eval_sudoku_ar.py \\
      --model meta-llama/Meta-Llama-3-8B-Instruct \\
      --few-shot 0 --batch-size 32 --output-dir results/llama3_0shot

  # DeepSeek-Math, 5-shot
  python eval_sudoku_ar.py \\
      --model deepseek-ai/deepseek-math-7b-instruct \\
      --few-shot 5 --batch-size 16 --output-dir results/deepseek_5shot

  # Via shell scripts
  bash run_llama3_0shot.sh
  bash run_llama3_5shot.sh
  bash run_deepseek_0shot.sh
  bash run_deepseek_5shot.sh
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ===============================================================================
# 1.  Types
# ===============================================================================

class InferenceMode(str, Enum):
    ZERO_SHOT = "zero_shot"
    FEW_SHOT  = "few_shot"


@dataclass
class Problem:
    problem_id: str
    grid:       list[list[Any]]   # None = blank cell
    solution:   list[list[int]]
    metadata:   dict = field(default_factory=dict)

    @property
    def rows(self) -> int:
        return len(self.grid)

    @property
    def cols(self) -> int:
        return len(self.grid[0]) if self.grid else 0


# ===============================================================================
# 2.  Prompt building  (identical to eval_sudoku.py)
# ===============================================================================

class PromptBuilder:
    """Serialises puzzles as Python list-of-lists (0 = blank)."""

    @staticmethod
    def grid_to_list(grid: list[list[Any]]) -> list[list[int]]:
        return [[0 if v is None else int(v) for v in row] for row in grid]

    @classmethod
    def build(
        cls,
        problem: Problem,
        mode: InferenceMode,
        few_shot_examples: list[Problem] | None = None,
    ) -> str:
        puzzle_as_list = cls.grid_to_list(problem.grid)

        few_shot_prefix = ""
        if mode == InferenceMode.FEW_SHOT and few_shot_examples:
            examples_block = ""
            for ex in few_shot_examples:
                ex_puzzle = cls.grid_to_list(ex.grid)
                examples_block += (
                    f"\nExample:\nPuzzle:\n{ex_puzzle}\n\n"
                    f"SOLUTION (follow the solution structure above)\n{ex.solution}\n"
                )
            few_shot_prefix = (
                f"Here are some solved examples for reference:{examples_block}\n"
                "Now solve the following:\n"
            )

        rows = ",".join(f"[row{i+1}]" for i in range(problem.rows))
        return (
            f"{few_shot_prefix}"
            "Solve the following Sudoku puzzle.\n"
            f"The following grid is a {problem.rows}x{problem.cols} list of lists. "
            "Empty cells are represented by 0. \n"
            "INSTRUCTION: Fill only the cells containing 0. Preserving the initial "
            "values is mandatory while following all Sudoku rules.\n"
            f"\nPuzzle:\n{puzzle_as_list}\n\n"
            f"Return ONLY one solution, NO additional words, NO code, NO thinking "
            f"steps or similar ONLY ONE solution in {problem.rows}x{problem.cols} "
            "list of lists completely solved.\n"
            "\nSOLUTION STRUCTURE\n"
            f"[{rows}]\n"
            "\nSOLUTION (follow the solution structure above)\n"
        )


class GridParser:
    """Parse a model's text output back into a 2-D list."""

    @staticmethod
    def parse(text: str, expected_rows: int, expected_cols: int) -> list[list[Any]] | None:
        candidates = []
        start = 0
        while True:
            pos = text.find("[[", start)
            if pos == -1:
                break
            end = text.find("]]", pos)
            if end == -1:
                break
            candidates.append(text[pos : end + 2])
            start = pos + 1

        last_valid = None
        for candidate in candidates:
            try:
                grid = json.loads(candidate)
                if (
                    len(grid) == expected_rows
                    and all(len(r) == expected_cols for r in grid)
                ):
                    last_valid = grid
            except Exception:
                continue
        return last_valid


def render_result(
    problem: Problem,
    predicted: list[list[Any]] | None,
    show_diff: bool = True,
) -> str:
    """Render puzzle and prediction side by side; mark wrong cells with *."""
    TOP = "┌───────┬───────┬───────┐"
    MID = "├───────┼───────┼───────┤"
    BOT = "└───────┴───────┴───────┘"
    SEP = "   "

    def row_str(row, solution_row=None, pred_row=None):
        cells = []
        for c, val in enumerate(row):
            ch = "·" if (val is None or val == 0) else str(val)
            if show_diff and solution_row is not None and pred_row is not None:
                sv = solution_row[c] if c < len(solution_row) else None
                pv = pred_row[c]     if pred_row and c < len(pred_row) else None
                if pv is not None and sv is not None and pv != sv:
                    ch = "*"
            cells.append(ch)
        return "│ {0} {1} {2} │ {3} {4} {5} │ {6} {7} {8} │".format(*cells)

    puzzle   = problem.grid
    solution = problem.solution

    if predicted is None:
        pred_label = "PREDICTION  (parse failed)"
        pred_grid  = [[0] * 9 for _ in range(9)]
    elif show_diff:
        pred_label = "PREDICTION  (* = wrong)"
        pred_grid  = predicted
    else:
        pred_label = "PREDICTION"
        pred_grid  = predicted

    header = f"  {'PUZZLE':<25}{SEP}{pred_label}"
    lines  = [header]

    for r in range(min(9, problem.rows)):
        if r == 0:
            lines.append(f"  {TOP}{SEP}{TOP}")
        elif r % 3 == 0:
            lines.append(f"  {MID}{SEP}{MID}")

        p_row = puzzle[r]    if r < len(puzzle)    else [0] * 9
        s_row = solution[r]  if r < len(solution)  else None
        d_row = pred_grid[r] if r < len(pred_grid) else None

        puzzle_line = row_str(p_row)
        pred_line   = row_str(d_row or [0] * 9, solution_row=s_row, pred_row=d_row)
        lines.append(f"  {puzzle_line}{SEP}{pred_line}")

    lines.append(f"  {BOT}{SEP}{BOT}")
    return "\n".join(lines)


# ===============================================================================
# 3.  Dataset  (identical to eval_sudoku.py)
# ===============================================================================

class DifficultyFilter:
    VALID_TIERS = {"easy", "medium", "hard"}

    def __init__(
        self,
        tier:                    str                | None = None,
        missing_cells:           tuple[int, int]    | None = None,
        given_ratio:             tuple[float, float] | None = None,
        naked_singles_count:     tuple[int, int]    | None = None,
        hidden_singles_count:    tuple[int, int]    | None = None,
        initial_resolution_rate: tuple[float, float] | None = None,
    ):
        if tier is not None and tier not in self.VALID_TIERS:
            raise ValueError(f"tier must be one of {self.VALID_TIERS}, got {tier!r}")
        self.tier                    = tier
        self.missing_cells           = missing_cells
        self.given_ratio             = given_ratio
        self.naked_singles_count     = naked_singles_count
        self.hidden_singles_count    = hidden_singles_count
        self.initial_resolution_rate = initial_resolution_rate

    @classmethod
    def from_preset(cls, difficulty: str) -> "DifficultyFilter":
        return cls(tier=str(difficulty).lower())

    def matches(self, meta: dict) -> bool:
        if self.tier is not None and meta.get("difficulty_tier") != self.tier:
            return False
        for key, rng in [
            ("missing_cells",           self.missing_cells),
            ("given_ratio",             self.given_ratio),
            ("naked_singles_count",     self.naked_singles_count),
            ("hidden_singles_count",    self.hidden_singles_count),
            ("initial_resolution_rate", self.initial_resolution_rate),
        ]:
            if rng is not None and key in meta:
                lo, hi = rng
                if not (lo <= meta[key] <= hi):
                    return False
        return True


class NineGrid:
    """Stream GridCorpus from HuggingFace: beta3/GridCorpus_9M_Sudoku_Puzzles_Enriched."""

    HF_REPO   = "beta3/GridCorpus_9M_Sudoku_Puzzles_Enriched"
    HF_SUBSET = "gridcorpus"
    HF_SPLIT  = "data"

    _META_COLS = [
        "missing_cells", "given_cells", "given_ratio",
        "row_given_counts", "col_given_counts",
        "naked_singles_count", "hidden_singles_count",
        "initial_resolution_rate", "difficulty_tier",
        "requires_backtrack", "backtrack_depth",
        "constraint_propagation_steps",
    ]

    def __init__(
        self,
        n_samples: int | None = None,
        difficulty: DifficultyFilter | None = None,
        few_shot_strategy: str = "easiest",
        seed: int = 42,
        streaming: bool = True,
    ):
        self._rng               = np.random.default_rng(seed)
        self._few_shot_strategy = few_shot_strategy
        self._problems          = self._load(n_samples, difficulty, streaming)
        if not self._problems:
            raise ValueError(f"No problems loaded — check the difficulty filter: {difficulty}")

    def __len__(self)  -> int:  return len(self._problems)
    def __iter__(self):         return iter(self._problems)

    def few_shot_examples(self, k: int = 3) -> list[Problem]:
        k = min(k, len(self._problems))
        if self._few_shot_strategy == "easiest":
            return sorted(self._problems,
                          key=lambda p: p.metadata.get("missing_cells", 0))[:k]
        elif self._few_shot_strategy == "hardest":
            return sorted(self._problems,
                          key=lambda p: p.metadata.get("missing_cells", 0),
                          reverse=True)[:k]
        else:
            idx = self._rng.choice(len(self._problems), size=k, replace=False)
            return [self._problems[i] for i in idx]

    def _load(self, n_samples, difficulty, streaming) -> list[Problem]:
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError("pip install datasets") from e

        print(f"  [NineGrid] Streaming from HuggingFace: {self.HF_REPO}")
        ds = load_dataset(self.HF_REPO, self.HF_SUBSET, split=self.HF_SPLIT,
                          streaming=streaming)

        problems: list[Problem] = []
        for idx, row in enumerate(ds):
            meta = {col: row[col] for col in self._META_COLS if col in row}
            if difficulty is not None and not difficulty.matches(meta):
                continue
            puzzle_grid   = self._coerce_grid(row["puzzle_grid"])
            solution_grid = self._coerce_grid(row["solution_grid"])
            grid = [[v if v != 0 else None for v in r] for r in puzzle_grid]
            problems.append(Problem(
                problem_id=str(row.get("id", idx)),
                grid=grid, solution=solution_grid, metadata=meta,
            ))
            if n_samples is not None and len(problems) >= n_samples:
                break
        return problems

    @staticmethod
    def _coerce_grid(value: Any) -> list[list[int]]:
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            return ast.literal_eval(value)
        raise ValueError(f"Cannot parse grid from {type(value)}: {value!r}")


# ===============================================================================
# 4.  Metrics  (identical to eval_sudoku.py)
# ===============================================================================

def compute_metrics(problem: Problem, predicted: list[list[Any]] | None) -> dict:
    if predicted is None:
        return dict(cell_acc=0.0, board_acc=0.0, row_acc=0.0, col_acc=0.0,
                    parse_fail=True)

    rows, cols = problem.rows, problem.cols
    blank_total = blank_correct = 0
    row_ok = [True] * rows
    col_ok = [True] * cols

    for r in range(rows):
        for c in range(cols):
            if problem.grid[r][c] is None:
                blank_total += 1
                sv = problem.solution[r][c] if r < len(problem.solution) and c < len(problem.solution[r]) else None
                pv = predicted[r][c]        if r < len(predicted)        and c < len(predicted[r])        else None
                if pv is not None and pv == sv:
                    blank_correct += 1
                else:
                    row_ok[r] = False
                    col_ok[c] = False

    return dict(
        cell_acc  = blank_correct / blank_total if blank_total else 1.0,
        board_acc = 1.0 if all(row_ok) else 0.0,
        row_acc   = float(sum(row_ok)) / rows,
        col_acc   = float(sum(col_ok)) / cols,
        parse_fail= False,
    )


def aggregate_metrics(results: list[dict]) -> dict:
    n = len(results)
    if n == 0:
        return {}
    n_fail = sum(r["parse_fail"] for r in results)
    return {
        "n_total":        n,
        "n_parse_fail":   n_fail,
        "pct_parse_fail": round(100.0 * n_fail / n, 3),
        "cell_acc":       float(np.mean([r["cell_acc"]  for r in results])),
        "board_acc":      float(np.mean([r["board_acc"] for r in results])),
        "row_acc":        float(np.mean([r["row_acc"]   for r in results])),
        "col_acc":        float(np.mean([r["col_acc"]   for r in results])),
    }


# ===============================================================================
# 5.  Model loading & generation
# ===============================================================================

# Models that need trust_remote_code=True for their tokenizer / config.
_TRUST_REMOTE_CODE_MODELS = {
    "deepseek-ai/deepseek-math-7b-instruct",
    "deepseek-ai/deepseek-math-7b-base",
}

def _needs_trust(model_path: str) -> bool:
    return any(k in model_path.lower() for k in
               ["deepseek-math", "deepseek_math"])


def load_model_and_tokenizer(model_path: str, device: str):
    trust = _needs_trust(model_path)
    print(f"[eval_sudoku_ar] Loading model : {model_path}")
    print(f"[eval_sudoku_ar] trust_remote_code={trust}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=trust,
    )

    # Batched left-padded generation requires a pad token.
    # Llama-3 ships without one; reuse eos_token as pad (common practice).
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print(f"[eval_sudoku_ar] pad_token set to eos_token "
              f"(id={tokenizer.pad_token_id})")

    # Left-pad so the prompt is right-aligned — necessary for correct
    # attention in batched generation.
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=trust,
        torch_dtype=torch.bfloat16,
        device_map=device,        # handles multi-GPU / CPU offload automatically
    ).eval()

    return model, tokenizer


def build_chat_prompt(
    tokenizer,
    problem: Problem,
    mode: InferenceMode,
    few_shot_examples: list[Problem] | None,
) -> str:
    raw = PromptBuilder.build(problem, mode, few_shot_examples)
    # apply_chat_template handles model-specific formatting (system tokens,
    # [INST] tags, etc.) automatically from the tokenizer config.
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": raw}],
        add_generation_prompt=True,
        tokenize=False,
    )


def run_generation_batch(
    model,
    tokenizer,
    prompt_texts: list[str],
    device: str,
    max_new_tokens: int,
) -> list[str]:
    encoded = tokenizer(
        prompt_texts,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )
    input_ids      = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)
    prompt_len     = input_ids.shape[1]

    with torch.no_grad():
        out = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # greedy — matches LLaDA's temperature=0
            temperature=None,         # suppress HF warning when do_sample=False
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = out[:, prompt_len:]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)


# ===============================================================================
# 6.  Evaluation loop
# ===============================================================================

def evaluate_difficulty(
    *,
    model,
    tokenizer,
    problems: list[Problem],
    difficulty_name: str,
    device: str,
    batch_size: int,
    max_new_tokens: int,
    mode: InferenceMode,
    few_shot_examples: list[Problem] | None,
    n_success_samples: int = 5,
    n_failure_samples: int = 10,
) -> dict:
    results:   list[dict] = []
    successes: list[dict] = []
    failures:  list[dict] = []

    n_batches = (len(problems) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(n_batches), desc=f"  [{difficulty_name}]"):
        batch_problems = problems[batch_idx * batch_size : (batch_idx + 1) * batch_size]

        prompt_texts = [
            build_chat_prompt(tokenizer, prob, mode, few_shot_examples)
            for prob in batch_problems
        ]

        try:
            generated_texts = run_generation_batch(
                model, tokenizer, prompt_texts, device, max_new_tokens,
            )
        except Exception as exc:
            print(
                f"\n[eval_sudoku_ar] Batch {batch_idx} failed ({exc}); "
                "recording as parse failures.",
                file=sys.stderr,
            )
            for prob in batch_problems:
                results.append(dict(
                    cell_acc=0.0, board_acc=0.0, row_acc=0.0, col_acc=0.0,
                    parse_fail=True, generated_text="<BATCH_ERROR>",
                    problem_id=prob.problem_id,
                ))
            torch.cuda.empty_cache()
            continue

        for i, (prob, gen_text) in enumerate(zip(batch_problems, generated_texts)):
            predicted = GridParser.parse(gen_text, prob.rows, prob.cols)
            m         = compute_metrics(prob, predicted)
            m["generated_text"] = gen_text
            m["problem_id"]     = prob.problem_id
            results.append(m)

            sample = {
                "problem_id": prob.problem_id,
                "prompt":     prompt_texts[i],
                "generated":  gen_text,
                "parse_fail": m["parse_fail"],
                "metrics":    {k: v for k, v in m.items()
                               if k not in ("generated_text", "problem_id", "parse_fail")},
                "render":     render_result(prob, predicted),
            }
            if not m["parse_fail"] and m["board_acc"] == 1.0:
                if len(successes) < n_success_samples:
                    successes.append(sample)
            else:
                if len(failures) < n_failure_samples:
                    failures.append(sample)

        # ── Running stats after every batch ───────────────────────────────
        n_done    = len(results)
        n_fail    = sum(r["parse_fail"] for r in results)
        cell_acc  = float(np.mean([r["cell_acc"]  for r in results]))
        board_acc = float(np.mean([r["board_acc"] for r in results]))
        row_acc   = float(np.mean([r["row_acc"]   for r in results]))
        col_acc   = float(np.mean([r["col_acc"]   for r in results]))
        print(
            f"\n  [{difficulty_name}] "
            f"n={n_done:>6,}  "
            f"parse_fail={n_fail:>5,} ({100*n_fail/n_done:4.1f}%)  "
            f"cell={cell_acc:.4f}  board={board_acc:.4f}  "
            f"row={row_acc:.4f}  col={col_acc:.4f}",
            flush=True,
        )

        torch.cuda.empty_cache()

    return {
        "difficulty": difficulty_name,
        "aggregate":  aggregate_metrics(results),
        "successes":  successes,
        "failures":   failures,
        "per_sample": results,
    }


# ===============================================================================
# 7.  Saving  (identical format to eval_sudoku.py for easy comparison)
# ===============================================================================

def save_results(result: dict, out_prefix: Path, args: argparse.Namespace) -> None:
    prefix = str(out_prefix)

    with open(prefix + ".summary.json", "w") as f:
        json.dump({"config": vars(args), "difficulty": result["difficulty"],
                   "aggregate": result["aggregate"]}, f, indent=2)

    with open(prefix + ".per_sample.jsonl", "w") as f:
        for s in result["per_sample"]:
            f.write(json.dumps({k: v for k, v in s.items() if k != "prompt"},
                               default=str) + "\n")

    with open(prefix + ".samples.json", "w") as f:
        json.dump({"successes": result["successes"],
                   "failures":  result["failures"]}, f, indent=2, default=str)

    shot_tag = f"{args.few_shot}-shot"
    with open(prefix + ".samples.txt", "w") as f:
        _write_section(f, "SUCCESSES", result["difficulty"], shot_tag, result["successes"])
        _write_section(f, "FAILURES",  result["difficulty"], shot_tag, result["failures"])

    print(f"    -> {prefix}.summary.json")
    print(f"    -> {prefix}.per_sample.jsonl")
    print(f"    -> {prefix}.samples.json")
    print(f"    -> {prefix}.samples.txt")


def _write_section(f, label, diff, shot_tag, samples):
    bar = "=" * 70
    f.write(f"{bar}\n{label}  [{diff}  {shot_tag}]  ({len(samples)} samples)\n{bar}\n\n")
    for s in samples:
        f.write(f"Problem ID : {s['problem_id']}\n")
        f.write(f"Parse fail : {s['parse_fail']}\n")
        f.write(f"Metrics    : {s['metrics']}\n\n")
        f.write(s.get("render", "(no render)") + "\n\n")
        f.write("Generated output:\n")
        f.write(s.get("generated", "") + "\n")
        f.write("-" * 40 + "\n\n")


# ===============================================================================
# 8.  CLI
# ===============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate an AR HuggingFace model on GridCorpus Sudoku.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model
    p.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct",
                   help="HuggingFace model ID or local path.")
    p.add_argument("--device", default="cuda",
                   help="Passed to device_map. Use 'auto' to spread across GPUs.")
    # Dataset
    p.add_argument("--difficulty", nargs="+", default=["easy", "hard"],
                   choices=["easy", "medium", "hard"])
    p.add_argument("--n-samples",   type=int, default=20_000,
                   help="Max problems per difficulty.")
    p.add_argument("--no-streaming", action="store_true",
                   help="Download full dataset instead of streaming.")
    p.add_argument("--seed", type=int, default=42)
    # Generation
    p.add_argument("--batch-size",    type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=512,
                   help="Maximum tokens to generate per prompt. "
                        "A 9x9 grid serialised as a list takes ~180 tokens; "
                        "512 gives comfortable headroom for both 0-shot and 5-shot.")
    # Few-shot
    p.add_argument("--few-shot", type=int, default=0,
                   help="Number of in-context examples (0 = zero-shot).")
    # Output
    p.add_argument("--output-dir",        default="results")
    p.add_argument("--n-success-samples", type=int, default=5)
    p.add_argument("--n-failure-samples", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Derive a short tag from the model name for filenames, e.g.
    #   meta-llama/Meta-Llama-3-8B-Instruct  -> llama-3-8b-instruct
    model_tag = args.model.split("/")[-1].lower().replace("_", "-")
    shot_tag  = f"{args.few_shot}shot"

    model, tokenizer = load_model_and_tokenizer(args.model, args.device)
    mode = InferenceMode.FEW_SHOT if args.few_shot > 0 else InferenceMode.ZERO_SHOT

    all_aggregates: dict[str, dict] = {}
    t_total = time.time()

    for diff_name in args.difficulty:
        print(f"\n{'─' * 70}")
        print(f"  Model      : {args.model}")
        print(f"  Difficulty : {diff_name.upper()}    few-shot : {args.few_shot}")
        print(f"{'─' * 70}")

        diff_filter = DifficultyFilter.from_preset(diff_name)

        dataset      = NineGrid(
            n_samples=args.n_samples + (args.few_shot or 0) + 100,
            difficulty=diff_filter,
            seed=args.seed,
            streaming=not args.no_streaming,
        )
        all_problems = list(dataset)
        print(f"  Loaded {len(all_problems):,} [{diff_name}] problems.")

        few_shot_examples: list[Problem] | None = None
        few_shot_ids: set[str] = set()
        if args.few_shot > 0:
            few_shot_examples = dataset.few_shot_examples(k=args.few_shot)
            few_shot_ids      = {ex.problem_id for ex in few_shot_examples}
            print(f"  Reserving {len(few_shot_examples)} few-shot examples.")

        test_pool = [p for p in all_problems if p.problem_id not in few_shot_ids]
        if len(test_pool) > args.n_samples:
            rng     = np.random.default_rng(args.seed)
            indices = rng.choice(len(test_pool), size=args.n_samples, replace=False)
            test_problems = [test_pool[int(i)] for i in sorted(indices)]
        else:
            test_problems = test_pool
        print(f"  Test set   : {len(test_problems):,} problems")

        t0     = time.time()
        result = evaluate_difficulty(
            model=model, tokenizer=tokenizer,
            problems=test_problems,
            difficulty_name=diff_name,
            device=args.device,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            mode=mode,
            few_shot_examples=few_shot_examples,
            n_success_samples=args.n_success_samples,
            n_failure_samples=args.n_failure_samples,
        )
        elapsed = time.time() - t0

        agg = result["aggregate"]
        all_aggregates[diff_name] = agg

        print(f"\n  Results -- {diff_name} ({model_tag}, {shot_tag})")
        print(f"  {'Evaluated':<20}: {agg['n_total']:>10,}")
        print(f"  {'Parse failures':<20}: {agg['n_parse_fail']:>10,}  ({agg['pct_parse_fail']:.1f}%)")
        print(f"  {'Cell accuracy':<20}: {agg['cell_acc']:>10.4f}")
        print(f"  {'Board accuracy':<20}: {agg['board_acc']:>10.4f}")
        print(f"  {'Row accuracy':<20}: {agg['row_acc']:>10.4f}")
        print(f"  {'Col accuracy':<20}: {agg['col_acc']:>10.4f}")
        print(f"  {'Wall time':<20}: {elapsed:>10.1f}s")

        print(f"\n  Saving to {output_dir}/")
        out_prefix = output_dir / f"{model_tag}_{diff_name}_{shot_tag}"
        save_results(result, out_prefix, args)

    combined_path = output_dir / f"{model_tag}_combined_{shot_tag}.json"
    with open(combined_path, "w") as f:
        json.dump({
            "config":       vars(args),
            "model_tag":    model_tag,
            "shot_tag":     shot_tag,
            "total_time_s": round(time.time() - t_total, 1),
            "results":      all_aggregates,
        }, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"  Combined summary -> {combined_path}")
    print(f"  Total wall time  : {time.time() - t_total:.1f}s")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()

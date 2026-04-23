#!/usr/bin/env python3
"""
eval_kakuro.py  —  Evaluate LLaDA-8B-Instruct on Sudoku (NineGrid) benchmarks.

Self-contained: all types, prompt building, grid parsing, and dataset loading
are inlined here so the script does not depend on the package structure of
prompt.py / dataset.py (which use relative imports from .types / .eval_types).

The only local dependency is generate.py, which has no relative imports.

Metrics reported per difficulty
────────────────────────────────
  cell_acc   : fraction of originally-blank cells predicted correctly
  board_acc  : fraction of boards solved perfectly (all blank cells correct)
  row_acc    : fraction of rows solved perfectly   (averaged per board, then across boards)
  col_acc    : fraction of columns solved perfectly (same averaging)
  parse_fail : count / percent of responses that could not be parsed as a grid

Usage examples
────────────────────────────────
  # Zero-shot, 20 k samples per difficulty, batch size 64
  python eval_sudoku.py --parquet ninegrid.parquet --few-shot 0 --batch-size 64

  # 5-shot
  python eval_sudoku.py --parquet ninegrid.parquet --few-shot 5 --batch-size 64

  # Run via the provided shell scripts
  bash run_0shot.sh
  bash run_5shot.sh
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

import re
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

# generate.py has no relative imports — import it directly.
try:
    from generate import generate
except ImportError as exc:
    sys.exit(
        f"[eval_sudoku] Cannot import generate.py: {exc}\n"
        "Place eval_sudoku.py in the same directory as generate.py and try again."
    )

VERBOSE=True

# ===============================================================================
# 1.  Types  (replaces .types / .eval_types)
# ===============================================================================

class InferenceMode(str, Enum):
    ZERO_SHOT = "zero_shot"
    FEW_SHOT  = "few_shot"


class Difficulty(str, Enum):
    EASY   = "easy"
    MEDIUM = "medium"
    HARD   = "hard"


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
# 2.  Prompt building  (from prompt.py, relative imports removed)
# ===============================================================================

class PromptBuilder:
    """Serialises Kakuro puzzles as Python list-of-lists."""

    @staticmethod
    def grid_to_list(grid: list[list[Any]]) -> list[list[Any]]:
        out = []
        for row in grid:
            new_row = []
            for v in row:
                if v is None:
                    new_row.append(-1)
                elif isinstance(v, tuple):
                    new_row.append((int(v[0]), int(v[1])))
                else:
                    new_row.append(int(v))
            out.append(new_row)
        return out

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
                    f"SOLUTION:\n{ex.solution}\n"
                )
            few_shot_prefix = (
                f"Here are some solved examples for reference:{examples_block}\n"
                "Now solve the following:\n"
            )

        rows = ",".join(f"[row{i+1}]" for i in range(problem.rows))

        return (
            f"{few_shot_prefix}"
            "Solve the following Kakuro puzzle.\n\n"
            f"The grid is a {problem.rows}x{problem.cols} list of lists.\n\n"
            "Cell meanings:\n"
            "- 0 = black cell (do not fill)\n"
            "- (A, D) = clue cell where:\n"
            "    A = sum of the ACROSS run (to the right)\n"
            "    D = sum of the DOWN run (below)\n"
            "- Note: Cells can be numbers or tuples\n"
            "- Empty cells to fill are represented by -1 in the puzzle but should be filled with digits 1-9\n\n"
            "KAKURO RULES:\n"
            "- Fill each empty cell (marked with -1) with a digit 1-9\n"
            "- Numbers in each each horizontal run must sum to the across clue, i.e., number A for row ending in tuple (A,D) \n"
            "- Numbers in each each vertical run must sum to the down clue, i.e., number D for column ending in tuple (A,D) \n"
            "- Leave 0s as unfilled\n"
            "- No repeated digits within a run\n\n"
            "IMPORTANT:\n"
            "- Do NOT modify black cells or clue cells\n"
            "- Only fill empty cells\n\n"
            f"Puzzle:\n{puzzle_as_list}\n\n"
            "Return ONLY the fully solved grid as a list of lists.\n"
            "NO explanations, NO extra text.\n\n"
            "SOLUTION STRUCTURE:\n"
            f"[{rows}]\n\n"
            "SOLUTION:\n"
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
                # grid = json.loads(candidate)
                grid = ast.literal_eval(candidate)
                if (
                    len(grid) == expected_rows
                    and all(len(r) == expected_cols for r in grid)
                ):
                    last_valid = grid
            
            except ValueError as je:
                print(f"Decoding response using json failed for candidate {candidate}")
                continue
            except Exception as e:
                if VERBOSE:
                    print(f"Got exception {e} when parsing grid candidate {candidate}")
                continue
        
        if VERBOSE and last_valid is None:
            print("No valid candidates")
        return last_valid

def render_result(
    problem: Problem,
    predicted: list[list[Any]] | None,
    show_diff: bool = True,
) -> str:
    """
    Render Kakuro board nicely with centered cells.
    """

    puzzle = problem.grid
    solution = problem.solution

    rows, cols = problem.rows, problem.cols

    # fallback prediction
    if predicted is None:
        pred_label = "PREDICTION (parse failed)"
        pred = [[None] * cols for _ in range(rows)]
    else:
        pred_label = "PREDICTION (* = wrong)" if show_diff else "PREDICTION"
        pred = predicted

    # ---- cell formatter ----
    def fmt(cell, width=7):
        if cell == 0:
            return "0".center(width)
        elif isinstance(cell, tuple):
            a, d = cell
            return f"({a},{d})".center(width)
        elif cell is None:
            return ".".center(width)
        else:
            return str(cell).center(width)

    def fmt_pred(r, c):
        pv = pred[r][c] if r < len(pred) and c < len(pred[r]) else None
        sv = solution[r][c] if r < len(solution) and c < len(solution[r]) else None

        if puzzle[r][c] is None:
            if pv is None:
                return ".".center(7)
            if show_diff and sv is not None and pv != sv:
                return "*".center(7)
            return str(pv).center(7)
        else:
            return fmt(puzzle[r][c])

    # ---- horizontal border ----
    def border():
        return "+" + "+".join(["-" * 7 for _ in range(cols)]) + "+"

    lines = []
    lines.append(f"{'PUZZLE':^{cols*8}}   {pred_label}")

    lines.append(border() + "   " + border())

    for r in range(rows):
        left = "|".join(fmt(solution[r][c]) for c in range(cols))
        right = "|".join(fmt_pred(r, c) for c in range(cols))

        lines.append(f"|{left}|   |{right}|")
        lines.append(border() + "   " + border())

    return "\n".join(lines)


# ===============================================================================
# 3.  Dataset  (from dataset.py, relative imports removed)
# ===============================================================================

class DifficultyFilter:
    """
    Filter Kakuro_mania problems using the dataset's numeric difficulty column
    plus optional metadata bounds.

    Kakuro_mania columns include:
      - difficulty (float)
      - min_value, max_value (int)
      - num_entry_squares (int)
      - search_space_size (string)
      - brute_force_size (int)
      - is_exclusive (bool)
    """

    def __init__(
        self,
        difficulty: tuple[float, float] | None = None,
        min_value: tuple[int, int] | None = None,
        max_value: tuple[int, int] | None = None,
        num_entry_squares: tuple[int, int] | None = None,
        brute_force_size: tuple[int, int] | None = None,
        is_exclusive: bool | None = None,
    ):
        self.difficulty = difficulty
        self.min_value = min_value
        self.max_value = max_value
        self.num_entry_squares = num_entry_squares
        self.brute_force_size = brute_force_size
        self.is_exclusive = is_exclusive

    @classmethod
    def from_preset(
                    cls, 
                    level: str,
                    *,
                    is_exclusive:bool |None = None,
                    ) -> "DifficultyFilter":
        """
        A convenience preset over the numeric difficulty score.

        These ranges are heuristic; adjust after inspecting the dataset
        distribution in your experiments.
        """
        level = str(level).lower()
        presets = {
            "easy":   (0.00, 0.40),
            "medium": (0.40, 0.70),
            "hard":   (0.70, 1.00),
        }
        if level not in presets:
            raise ValueError(f"Unknown preset {level!r}; expected one of {set(presets)}")
        return cls(
                    difficulty=presets[level],
                    is_exclusive=is_exclusive,
                )

    def matches(self, meta: dict[str, Any]) -> bool:
        numeric_checks = [
            ("difficulty", self.difficulty),
            ("min_value", self.min_value),
            ("max_value", self.max_value),
            ("num_entry_squares", self.num_entry_squares),
            ("brute_force_size", self.brute_force_size),
        ]
        for key, rng in numeric_checks:
            if rng is not None and key in meta:
                lo, hi = rng
                if not (lo <= meta[key] <= hi):
                    return False

        if self.is_exclusive is not None:
            if "is_exclusive" not in meta or meta["is_exclusive"] != self.is_exclusive:
                return False

        return True


class KakuroGrid:
    """
    Load the HuggingFace Kakuro dataset:
        Apaxdor/Kakuro_mania

    Dataset shape:
      - subset: "default"
      - split: "train"

    The puzzle/solution columns are row-wise strings such as:
      [0,0,(0,2),...] [0,(7,2),1,...] ...

    Cell semantics:
      - 0           -> black square
      - 1           -> fillable entry square in the puzzle
      - (down,across) -> clue square
    """

    name = "KakuroGrid"

    HF_REPO = "Apaxdor/Kakuro_mania"
    HF_SUBSET = "default"
    HF_SPLIT = "train"

    _META_COLS = [
        "seed",
        "difficulty",
        "min_value",
        "max_value",
        "num_entry_squares",
        "search_space_size",
        "brute_force_size",
        "is_exclusive",
    ]

    def __init__(
        self,
        n_samples: int | None = None,
        difficulty: DifficultyFilter | None = None,
        few_shot_strategy: str = "easiest",
        seed: int = 42,
        streaming: bool = True,
    ):
        self._rng = np.random.default_rng(seed)
        self._few_shot_strategy = few_shot_strategy
        self._problems = self._load(n_samples, difficulty, streaming)

        if not self._problems:
            raise ValueError(
                "No problems loaded - check the difficulty filter.\n"
                f"  Difficulty: {difficulty}"
            )

    def __len__(self) -> int:
        return len(self._problems)

    def __iter__(self):
        return iter(self._problems)

    def few_shot_examples(self, k: int = 3) -> list["Problem"]:
        k = min(k, len(self._problems))

        # Reuse the exact same pattern as NineGrid, but sort on numeric difficulty.
        if self._few_shot_strategy == "easiest":
            return sorted(
                self._problems,
                key=lambda p: p.metadata.get("difficulty", 0.0),
            )[:k]
        elif self._few_shot_strategy == "hardest":
            return sorted(
                self._problems,
                key=lambda p: p.metadata.get("difficulty", 0.0),
                reverse=True,
            )[:k]
        else:
            indices = self._rng.choice(len(self._problems), size=k, replace=False)
            return [self._problems[i] for i in indices]

    def _load(
        self,
        n_samples: int | None,
        difficulty: DifficultyFilter | None,
        streaming: bool,
    ) -> list["Problem"]:
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError(
                "Install the HuggingFace datasets library: pip install datasets"
            ) from e

        print(
            f"  [KakuroGrid] Loading from HuggingFace: {self.HF_REPO} "
            f"(subset={self.HF_SUBSET}, split={self.HF_SPLIT}, streaming={streaming})"
        )

        ds = load_dataset(
            self.HF_REPO,
            self.HF_SUBSET,
            split=self.HF_SPLIT,
            streaming=streaming,
        )

        problems: list[Problem] = []
        for idx, row in enumerate(ds):
            meta: dict[str, Any] = {
                col: row[col] for col in self._META_COLS if col in row
            }

            if difficulty is not None and not difficulty.matches(meta):
                continue

            puzzle_grid = self._coerce_grid(row["puzzle"])
            solution_grid = self._coerce_grid(row["solution"])

            # Convert the puzzle to a model-friendly grid:
            #   1 -> None (to be filled)
            #   0 -> 0 (black square)
            #   (a,b) -> clue tuple
            grid = self._normalize_puzzle_grid(puzzle_grid)

            problems.append(
                Problem(
                    problem_id=str(row.get("seed", idx)),
                    grid=grid,
                    solution=solution_grid,
                    metadata=meta,
                )
            )

            if n_samples is not None and len(problems) >= n_samples:
                break

        return problems

    @classmethod
    def _normalize_puzzle_grid(cls, puzzle_grid: list[list[Any]]) -> list[list[Any]]:
        normalized: list[list[Any]] = []
        for row in puzzle_grid:
            out_row = []
            for cell in row:
                if cell == 1:
                    out_row.append(None)   # fillable square
                else:
                    out_row.append(cell)   # 0 or clue tuple
            normalized.append(out_row)
        return normalized

    @classmethod
    def _coerce_grid(cls, value: Any) -> list[list[Any]]:
        """
        Parse Kakuro rows from either:
          - already-materialized nested lists, or
          - strings like:
              "[0,0,(0,2)] [0,(7,2),1]"
        """
        if isinstance(value, list):
            return value

        if not isinstance(value, str):
            raise ValueError(f"Cannot parse grid from {type(value)}: {value!r}")

        text = value.strip()

        # Case 1: already a real nested Python list string
        if text.startswith("[["):
            parsed = ast.literal_eval(text)
            if not isinstance(parsed, list):
                raise ValueError(f"Expected list grid, got: {type(parsed)}")
            return parsed

        # Case 2: dataset row format: "[... ] [... ] [... ]"
        row_strings = cls._split_row_strings(text)
        return [ast.literal_eval(r) for r in row_strings]

    @staticmethod
    def _split_row_strings(text: str) -> list[str]:
        """
        Extract top-level row lists from a string of the form:
            "[... ] [... ] [... ]"

        This is safer than naive splitting because cells may contain tuples.
        """
        rows: list[str] = []
        depth = 0
        start = None

        for i, ch in enumerate(text):
            if ch == "[":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0 and start is not None:
                    rows.append(text[start:i + 1])

        if not rows:
            raise ValueError(f"Could not parse Kakuro grid rows from: {text!r}")

        return rows

# ===============================================================================
# 4.  Metrics
# ===============================================================================

def compute_metrics(problem: Problem, predicted: list[list[Any]] | None) -> dict:
    """
    Per-sample metrics. Only blank cells (grid value = None) are scored.

    Returns
    -------
    cell_acc   : fraction of blank cells predicted correctly
    board_acc  : 1.0 iff every blank cell is correct
    row_acc    : fraction of rows where every blank cell is correct
    col_acc    : fraction of columns where every blank cell is correct
    parse_fail : True if predicted is None
    """
    if predicted is None:
        return dict(cell_acc=0.0, board_acc=0.0, row_acc=0.0, col_acc=0.0,
                    parse_fail=True)

    rows, cols = problem.rows, problem.cols
    puzzle     = problem.grid
    solution   = problem.solution

    blank_total   = 0
    blank_correct = 0
    row_ok        = [True] * rows
    col_ok        = [True] * cols
    # For Kakuro
    row_has_target = [False] * rows
    col_has_target = [False] * cols

    for r in range(rows):
        for c in range(cols):
            if puzzle[r][c] is None:
                # For Kakuro
                row_has_target[r] = True
                col_has_target[c] = True

                blank_total += 1
                sv = solution[r][c] if r < len(solution) and c < len(solution[r]) else None
                pv = predicted[r][c] if r < len(predicted) and c < len(predicted[r]) else None
                if pv is not None and pv == sv:
                    blank_correct += 1
                else:
                    row_ok[r] = False
                    col_ok[c] = False

    cell_acc = blank_correct / blank_total if blank_total else 1.0
    board_acc = 1.0 if blank_correct == blank_total else 0.0

    scored_rows = sum(row_has_target)
    scored_cols = sum(col_has_target)

    row_acc = (
        sum(ok for ok, has_target in zip(row_ok, row_has_target) if has_target) / scored_rows
        if scored_rows else 1.0
    )
    col_acc = (
        sum(ok for ok, has_target in zip(col_ok, col_has_target) if has_target) / scored_cols
        if scored_cols else 1.0
    )

    return dict(cell_acc=cell_acc, board_acc=board_acc,
                row_acc=row_acc, col_acc=col_acc, parse_fail=False)


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
# 5.  Model & generation
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
# 7.  Saving
# ===============================================================================

def save_results(result: dict, out_prefix: Path, args: argparse.Namespace) -> None:
    prefix = str(out_prefix)

    # Summary JSON
    summary_path = prefix + ".summary.json"
    with open(summary_path, "w") as f:
        json.dump({"config": vars(args), "difficulty": result["difficulty"],
                   "aggregate": result["aggregate"]}, f, indent=2)

    # Per-sample JSONL (no raw prompts to keep size down)
    per_sample_path = prefix + ".per_sample.jsonl"
    with open(per_sample_path, "w") as f:
        for s in result["per_sample"]:
            row = {k: v for k, v in s.items() if k != "prompt"}
            f.write(json.dumps(row, default=str) + "\n")

    # Sample renders — JSON
    samples_path = prefix + ".samples.json"
    with open(samples_path, "w") as f:
        json.dump({"successes": result["successes"],
                   "failures":  result["failures"]}, f, indent=2, default=str)

    # Sample renders — human-readable text
    txt_path = prefix + ".samples.txt"
    shot_tag = f"{args.few_shot}-shot"
    with open(txt_path, "w") as f:
        _write_section(f, "SUCCESSES", result["difficulty"], shot_tag, result["successes"])
        _write_section(f, "FAILURES",  result["difficulty"], shot_tag, result["failures"])

    print(f"    -> {summary_path}")
    print(f"    -> {per_sample_path}")
    print(f"    -> {samples_path}")
    print(f"    -> {txt_path}")


def _write_section(f, label: str, diff: str, shot_tag: str, samples: list[dict]) -> None:
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
        description="Evaluate an AR HuggingFace model on Apaxdor Kakuro_mania.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Dataset
    p.add_argument("--difficulty", nargs="+", default=["easy", "hard"],
                   choices=["easy", "medium", "hard"])
    p.add_argument("--n-samples",  type=int, default=10_000,
                   help="Max problems per difficulty (stops streaming early).")
    p.add_argument("--no-streaming", action="store_true",
                   help="Download the full dataset instead of streaming. "
                        "Faster if you plan multiple runs.")
    p.add_argument("--seed",       type=int, default=42)
    # Model
    p.add_argument("--model",   default="meta-llama/Meta-Llama-3-8B-Instruct")
    p.add_argument("--device",  default="cuda")

    # Generation
    p.add_argument("--batch-size",   type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=512,
                   help="Maximum tokens to generate per prompt. "
                        "A 9x9 grid serialised as a list takes ~180 tokens; "
                        "512 gives comfortable headroom for both 0-shot and 5-shot.")
    
    # Few-shot
    p.add_argument("--few-shot", type=int, default=0,
                   help="In-context examples prepended to each prompt (0 = zero-shot). "
                        "Examples are the easiest k puzzles; excluded from test set.")
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

        diff_filter = DifficultyFilter(
                                            difficulty=(0.4, 0.7),   # Medium
                                            is_exclusive=True
                                        )

        print("  Loading dataset ...")
        dataset      = KakuroGrid(
            n_samples=args.n_samples + (args.few_shot or 0) + 100,  # load a bit extra for few-shot headroom
            difficulty=diff_filter,
            seed=args.seed,
            streaming=not args.no_streaming,
        )
        all_problems = list(dataset)
        print(f"  Loaded {len(all_problems):,} [{diff_name}] problems.")

        # Few-shot examples chosen from the full pool before subsampling
        few_shot_examples: list[Problem] | None = None
        few_shot_ids: set[str] = set()
        if args.few_shot > 0:
            few_shot_examples = dataset.few_shot_examples(k=args.few_shot)
            few_shot_ids      = {ex.problem_id for ex in few_shot_examples}
            print(f"  Reserving {len(few_shot_examples)} few-shot examples "
                  f"(excluded from test set).")

        # Subsample test pool
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

        print(f"\n  Results -- {diff_name} ({shot_tag})")
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


    # Combined summary across all difficulties
    combined_path = output_dir / f"{model_tag}_combined_{shot_tag}.json"
    with open(combined_path, "w") as f:
        json.dump({
            "config":        vars(args),
            "model_tag":    model_tag,
            "shot_tag":      shot_tag,
            "total_time_s":  round(time.time() - t_total, 1),
            "results":       all_aggregates,
        }, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"  Combined summary -> {combined_path}")
    print(f"  Total wall time  : {time.time() - t_total:.1f}s")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
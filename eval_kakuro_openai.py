#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

load_dotenv()

VERBOSE = True


try:
    from generate import generate  # optional; retained for compatibility
except ImportError:
    generate = None


# ==============================================================================
# 1. Types
# ==============================================================================

class InferenceMode(str, Enum):
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"


class ModelBackend(str, Enum):
    HF = "hf"
    OPENAI = "openai"


@dataclass
class Problem:
    problem_id: str
    grid: list[list[Any]]
    solution: list[list[int]]
    metadata: dict = field(default_factory=dict)

    @property
    def rows(self) -> int:
        return len(self.grid)

    @property
    def cols(self) -> int:
        return len(self.grid[0]) if self.grid else 0


# ==============================================================================
# 2. Prompt building
# ==============================================================================

class PromptBuilder:
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

        rows = ",".join(f"[row{i + 1}]" for i in range(problem.rows))

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
            "- Fill each empty cell marked with -1 using a digit 1-9\n"
            "- Numbers in each horizontal run must sum to the across clue\n"
            "- Numbers in each vertical run must sum to the down clue\n"
            "- Leave 0s as unfilled\n"
            "- No repeated digits within a run\n\n"
            "IMPORTANT:\n"
            "- Do NOT modify black cells or clue cells\n"
            "- Only fill empty cells\n\n"
            f"Puzzle:\n{puzzle_as_list}\n\n"
            "Return ONLY the fully solved grid as a Python list of lists.\n"
            "NO explanations, NO markdown, NO extra text.\n\n"
            "SOLUTION STRUCTURE:\n"
            f"[{rows}]\n\n"
            "SOLUTION:\n"
        )


class GridParser:
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
            candidates.append(text[pos:end + 2])
            start = pos + 1

        last_valid = None

        for candidate in candidates:
            try:
                grid = ast.literal_eval(candidate)
                if (
                    isinstance(grid, list)
                    and len(grid) == expected_rows
                    and all(isinstance(r, list) and len(r) == expected_cols for r in grid)
                ):
                    last_valid = grid
            except Exception as e:
                if VERBOSE:
                    print(f"Parse exception {e} for candidate: {candidate[:200]}")
                continue

        if VERBOSE and last_valid is None:
            print("No valid grid candidates found.")

        return last_valid


def render_result(
    problem: Problem,
    predicted: list[list[Any]] | None,
    show_diff: bool = True,
) -> str:
    puzzle = problem.grid
    solution = problem.solution
    rows, cols = problem.rows, problem.cols

    if predicted is None:
        pred_label = "PREDICTION (parse failed)"
        pred = [[None] * cols for _ in range(rows)]
    else:
        pred_label = "PREDICTION (* = wrong)" if show_diff else "PREDICTION"
        pred = predicted

    def fmt(cell, width=7):
        if cell == 0:
            return "0".center(width)
        if isinstance(cell, tuple):
            a, d = cell
            return f"({a},{d})".center(width)
        if cell is None:
            return ".".center(width)
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

        return fmt(puzzle[r][c])

    def border():
        return "+" + "+".join(["-" * 7 for _ in range(cols)]) + "+"

    lines = []
    lines.append(f"{'SOLUTION':^{cols * 8}}   {pred_label}")
    lines.append(border() + "   " + border())

    for r in range(rows):
        left = "|".join(fmt(solution[r][c]) for c in range(cols))
        right = "|".join(fmt_pred(r, c) for c in range(cols))
        lines.append(f"|{left}|   |{right}|")
        lines.append(border() + "   " + border())

    return "\n".join(lines)


# ==============================================================================
# 3. Dataset
# ==============================================================================

class DifficultyFilter:
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
        is_exclusive: bool | None = None,
    ) -> "DifficultyFilter":
        level = str(level).lower()
        presets = {
            "easy": (0.00, 0.40),
            "medium": (0.40, 0.70),
            "hard": (0.70, 1.00),
        }

        if level not in presets:
            raise ValueError(f"Unknown difficulty preset {level!r}")

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
            if "is_exclusive" not in meta:
                return False
            if meta["is_exclusive"] != self.is_exclusive:
                return False

        return True


class KakuroGrid:
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
                "No problems loaded. Check your difficulty filter."
            )

    def __len__(self) -> int:
        return len(self._problems)

    def __iter__(self):
        return iter(self._problems)

    def few_shot_examples(self, k: int = 3) -> list[Problem]:
        k = min(k, len(self._problems))

        if self._few_shot_strategy == "easiest":
            return sorted(
                self._problems,
                key=lambda p: p.metadata.get("difficulty", 0.0),
            )[:k]

        if self._few_shot_strategy == "hardest":
            return sorted(
                self._problems,
                key=lambda p: p.metadata.get("difficulty", 0.0),
                reverse=True,
            )[:k]

        indices = self._rng.choice(len(self._problems), size=k, replace=False)
        return [self._problems[int(i)] for i in indices]

    def _load(
        self,
        n_samples: int | None,
        difficulty: DifficultyFilter | None,
        streaming: bool,
    ) -> list[Problem]:
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError(
                "Install datasets first: pip install datasets"
            ) from e

        print(
            f"  [KakuroGrid] Loading {self.HF_REPO} "
            f"subset={self.HF_SUBSET}, split={self.HF_SPLIT}, streaming={streaming}"
        )

        ds = load_dataset(
            self.HF_REPO,
            self.HF_SUBSET,
            split=self.HF_SPLIT,
            streaming=streaming,
        )

        problems: list[Problem] = []

        for idx, row in enumerate(ds):
            meta = {
                col: row[col]
                for col in self._META_COLS
                if col in row
            }

            if difficulty is not None and not difficulty.matches(meta):
                continue

            puzzle_grid = self._coerce_grid(row["puzzle"])
            solution_grid = self._coerce_grid(row["solution"])
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
        normalized = []

        for row in puzzle_grid:
            out_row = []
            for cell in row:
                if cell == 1:
                    out_row.append(None)
                else:
                    out_row.append(cell)
            normalized.append(out_row)

        return normalized

    @classmethod
    def _coerce_grid(cls, value: Any) -> list[list[Any]]:
        if isinstance(value, list):
            return value

        if not isinstance(value, str):
            raise ValueError(f"Cannot parse grid from {type(value)}: {value!r}")

        text = value.strip()

        if text.startswith("[["):
            parsed = ast.literal_eval(text)
            if not isinstance(parsed, list):
                raise ValueError(f"Expected list grid, got {type(parsed)}")
            return parsed

        row_strings = cls._split_row_strings(text)
        return [ast.literal_eval(r) for r in row_strings]

    @staticmethod
    def _split_row_strings(text: str) -> list[str]:
        rows = []
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
            raise ValueError(f"Could not parse Kakuro rows from: {text!r}")

        return rows


# ==============================================================================
# 4. Metrics
# ==============================================================================

def compute_metrics(problem: Problem, predicted: list[list[Any]] | None) -> dict:
    if predicted is None:
        return {
            "cell_acc": 0.0,
            "board_acc": 0.0,
            "row_acc": 0.0,
            "col_acc": 0.0,
            "parse_fail": True,
        }

    rows, cols = problem.rows, problem.cols
    puzzle = problem.grid
    solution = problem.solution

    blank_total = 0
    blank_correct = 0
    row_ok = [True] * rows
    col_ok = [True] * cols
    row_has_target = [False] * rows
    col_has_target = [False] * cols

    for r in range(rows):
        for c in range(cols):
            if puzzle[r][c] is None:
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
        sum(ok for ok, has in zip(row_ok, row_has_target) if has) / scored_rows
        if scored_rows else 1.0
    )

    col_acc = (
        sum(ok for ok, has in zip(col_ok, col_has_target) if has) / scored_cols
        if scored_cols else 1.0
    )

    return {
        "cell_acc": cell_acc,
        "board_acc": board_acc,
        "row_acc": row_acc,
        "col_acc": col_acc,
        "parse_fail": False,
    }


def aggregate_metrics(results: list[dict]) -> dict:
    n = len(results)

    if n == 0:
        return {}

    n_fail = sum(r["parse_fail"] for r in results)

    return {
        "n_total": n,
        "n_parse_fail": n_fail,
        "pct_parse_fail": round(100.0 * n_fail / n, 3),
        "cell_acc": float(np.mean([r["cell_acc"] for r in results])),
        "board_acc": float(np.mean([r["board_acc"] for r in results])),
        "row_acc": float(np.mean([r["row_acc"] for r in results])),
        "col_acc": float(np.mean([r["col_acc"] for r in results])),
    }


# ==============================================================================
# 5. Model loading and generation
# ==============================================================================

def _needs_trust(model_path: str) -> bool:
    return any(k in model_path.lower() for k in ["deepseek-math", "deepseek_math"])


def load_hf_model_and_tokenizer(model_path: str, device: str):
    trust = _needs_trust(model_path)

    print(f"[eval_kakuro] Loading HF model: {model_path}")
    print(f"[eval_kakuro] trust_remote_code={trust}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print(
            f"[eval_kakuro] pad_token set to eos_token "
            f"id={tokenizer.pad_token_id}"
        )

    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=trust,
        torch_dtype=torch.bfloat16,
        device_map=device,
    ).eval()

    return model, tokenizer


def load_openai_client(args):
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError("Install openai first: pip install openai") from e

    api_key = os.environ.get(args.api_key_env)

    if not api_key:
        raise ValueError(
            f"Missing API key. Set env var {args.api_key_env}. "
            "For vLLM, a dummy value is usually fine."
        )

    kwargs = {"api_key": api_key}

    if args.api_base:
        kwargs["base_url"] = args.api_base

    return OpenAI(**kwargs)


def build_raw_prompt(
    problem: Problem,
    mode: InferenceMode,
    few_shot_examples: list[Problem] | None,
) -> str:
    return PromptBuilder.build(problem, mode, few_shot_examples)


def build_hf_chat_prompt(
    tokenizer,
    problem: Problem,
    mode: InferenceMode,
    few_shot_examples: list[Problem] | None,
) -> str:
    raw = build_raw_prompt(problem, mode, few_shot_examples)

    return tokenizer.apply_chat_template(
        [{"role": "user", "content": raw}],
        add_generation_prompt=True,
        tokenize=False,
    )


def run_hf_generation_batch(
    model,
    tokenizer,
    prompt_texts: list[str],
    max_new_tokens: int,
) -> list[str]:
    encoded = tokenizer(
        prompt_texts,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)
    prompt_len = input_ids.shape[1]

    with torch.no_grad():
        out = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = out[:, prompt_len:]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)


def run_openai_generation_one(
    *,
    client,
    model_name: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    retries: int,
    retry_sleep: float,
) -> str:
    last_exc = None

    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_new_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content or ""

        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(retry_sleep * (2 ** attempt))

    raise RuntimeError(f"OpenAI-compatible generation failed: {last_exc}") from last_exc


def run_openai_generation_batch(
    *,
    client,
    model_name: str,
    prompt_texts: list[str],
    max_new_tokens: int,
    temperature: float,
    concurrency: int,
    retries: int,
    retry_sleep: float,
) -> list[str]:
    outputs: list[str | None] = [None] * len(prompt_texts)

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = {
            ex.submit(
                run_openai_generation_one,
                client=client,
                model_name=model_name,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                retries=retries,
                retry_sleep=retry_sleep,
            ): i
            for i, prompt in enumerate(prompt_texts)
        }

        for fut in as_completed(futures):
            i = futures[fut]
            outputs[i] = fut.result()

    return [o if o is not None else "" for o in outputs]


# ==============================================================================
# 6. Evaluation loop
# ==============================================================================

def evaluate_difficulty(
    *,
    backend: ModelBackend,
    model,
    tokenizer,
    client,
    model_name: str,
    problems: list[Problem],
    difficulty_name: str,
    batch_size: int,
    max_new_tokens: int,
    mode: InferenceMode,
    few_shot_examples: list[Problem] | None,
    temperature: float,
    openai_concurrency: int,
    openai_retries: int,
    openai_retry_sleep: float,
    n_success_samples: int = 5,
    n_failure_samples: int = 10,
) -> dict:
    results: list[dict] = []
    successes: list[dict] = []
    failures: list[dict] = []

    n_batches = (len(problems) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(n_batches), desc=f"  [{difficulty_name}]"):
        batch_problems = problems[
            batch_idx * batch_size:(batch_idx + 1) * batch_size
        ]

        if backend == ModelBackend.HF:
            prompt_texts = [
                build_hf_chat_prompt(tokenizer, prob, mode, few_shot_examples)
                for prob in batch_problems
            ]
        else:
            prompt_texts = [
                build_raw_prompt(prob, mode, few_shot_examples)
                for prob in batch_problems
            ]

        try:
            if backend == ModelBackend.HF:
                generated_texts = run_hf_generation_batch(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_texts=prompt_texts,
                    max_new_tokens=max_new_tokens,
                )
            else:
                generated_texts = run_openai_generation_batch(
                    client=client,
                    model_name=model_name,
                    prompt_texts=prompt_texts,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    concurrency=openai_concurrency,
                    retries=openai_retries,
                    retry_sleep=openai_retry_sleep,
                )

        except Exception as exc:
            print(
                f"\n[eval_kakuro] Batch {batch_idx} failed: {exc}. "
                "Recording as parse failures.",
                file=sys.stderr,
            )

            for prob in batch_problems:
                results.append(
                    {
                        "cell_acc": 0.0,
                        "board_acc": 0.0,
                        "row_acc": 0.0,
                        "col_acc": 0.0,
                        "parse_fail": True,
                        "generated_text": "<BATCH_ERROR>",
                        "problem_id": prob.problem_id,
                    }
                )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            continue

        for i, (prob, gen_text) in enumerate(zip(batch_problems, generated_texts)):
            predicted = GridParser.parse(gen_text, prob.rows, prob.cols)
            m = compute_metrics(prob, predicted)
            m["generated_text"] = gen_text
            m["problem_id"] = prob.problem_id
            results.append(m)

            sample = {
                "problem_id": prob.problem_id,
                "prompt": prompt_texts[i],
                "generated": gen_text,
                "parse_fail": m["parse_fail"],
                "metrics": {
                    k: v
                    for k, v in m.items()
                    if k not in ("generated_text", "problem_id", "parse_fail")
                },
                "render": render_result(prob, predicted),
            }

            if not m["parse_fail"] and m["board_acc"] == 1.0:
                if len(successes) < n_success_samples:
                    successes.append(sample)
            else:
                if len(failures) < n_failure_samples:
                    failures.append(sample)

        n_done = len(results)
        n_fail = sum(r["parse_fail"] for r in results)
        cell_acc = float(np.mean([r["cell_acc"] for r in results]))
        board_acc = float(np.mean([r["board_acc"] for r in results]))
        row_acc = float(np.mean([r["row_acc"] for r in results]))
        col_acc = float(np.mean([r["col_acc"] for r in results]))

        print(
            f"\n  [{difficulty_name}] "
            f"n={n_done:>6,}  "
            f"parse_fail={n_fail:>5,} ({100 * n_fail / n_done:4.1f}%)  "
            f"cell={cell_acc:.4f}  board={board_acc:.4f}  "
            f"row={row_acc:.4f}  col={col_acc:.4f}",
            flush=True,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "difficulty": difficulty_name,
        "aggregate": aggregate_metrics(results),
        "successes": successes,
        "failures": failures,
        "per_sample": results,
    }


# ==============================================================================
# 7. Saving
# ==============================================================================

def save_results(result: dict, out_prefix: Path, args: argparse.Namespace) -> None:
    prefix = str(out_prefix)

    summary_path = prefix + ".summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "config": vars(args),
                "difficulty": result["difficulty"],
                "aggregate": result["aggregate"],
            },
            f,
            indent=2,
        )

    per_sample_path = prefix + ".per_sample.jsonl"
    with open(per_sample_path, "w") as f:
        for s in result["per_sample"]:
            row = {k: v for k, v in s.items() if k != "prompt"}
            f.write(json.dumps(row, default=str) + "\n")

    samples_path = prefix + ".samples.json"
    with open(samples_path, "w") as f:
        json.dump(
            {
                "successes": result["successes"],
                "failures": result["failures"],
            },
            f,
            indent=2,
            default=str,
        )

    txt_path = prefix + ".samples.txt"
    shot_tag = f"{args.few_shot}-shot"

    with open(txt_path, "w") as f:
        _write_section(
            f,
            "SUCCESSES",
            result["difficulty"],
            shot_tag,
            result["successes"],
        )
        _write_section(
            f,
            "FAILURES",
            result["difficulty"],
            shot_tag,
            result["failures"],
        )

    print(f"    -> {summary_path}")
    print(f"    -> {per_sample_path}")
    print(f"    -> {samples_path}")
    print(f"    -> {txt_path}")


def _write_section(
    f,
    label: str,
    diff: str,
    shot_tag: str,
    samples: list[dict],
) -> None:
    bar = "=" * 70
    f.write(f"{bar}\n{label} [{diff} {shot_tag}] ({len(samples)} samples)\n{bar}\n\n")

    for s in samples:
        f.write(f"Problem ID : {s['problem_id']}\n")
        f.write(f"Parse fail : {s['parse_fail']}\n")
        f.write(f"Metrics    : {s['metrics']}\n\n")
        f.write(s.get("render", "(no render)") + "\n\n")
        f.write("Generated output:\n")
        f.write(s.get("generated", "") + "\n")
        f.write("-" * 40 + "\n\n")


# ==============================================================================
# 8. CLI
# ==============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate HF or OpenAI-compatible models on Apaxdor Kakuro_mania.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--difficulty",
        nargs="+",
        default=["easy", "hard"],
        choices=["easy", "medium", "hard"],
    )
    p.add_argument("--n-samples", type=int, default=10_000)
    p.add_argument("--no-streaming", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument(
        "--backend",
        choices=["hf", "openai"],
        default="hf",
        help="Use local HuggingFace generation or OpenAI-compatible chat completions.",
    )
    p.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    p.add_argument("--device", default="cuda")

    p.add_argument(
        "--api-base",
        default=None,
        help="OpenAI-compatible base URL. Example for vLLM: http://localhost:8000/v1",
    )
    p.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable containing API key.",
    )
    p.add_argument("--openai-concurrency", type=int, default=2)
    p.add_argument("--openai-retries", type=int, default=2)
    p.add_argument("--openai-retry-sleep", type=float, default=1.0)

    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)

    p.add_argument("--few-shot", type=int, default=0)
    p.add_argument("--few-shot-strategy", default="easiest",
                   choices=["easiest", "hardest", "random"])

    p.add_argument("--output-dir", default="results")
    p.add_argument("--n-success-samples", type=int, default=5)
    p.add_argument("--n-failure-samples", type=int, default=10)

    return p.parse_args()


# ==============================================================================
# 9. Main
# ==============================================================================

def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    backend = ModelBackend(args.backend)

    model_tag = (
        args.model.split("/")[-1]
        .lower()
        .replace("_", "-")
        .replace(":", "-")
    )
    shot_tag = f"{args.few_shot}shot"

    model = None
    tokenizer = None
    client = None

    if backend == ModelBackend.HF:
        model, tokenizer = load_hf_model_and_tokenizer(args.model, args.device)
    else:
        client = load_openai_client(args)

    mode = InferenceMode.FEW_SHOT if args.few_shot > 0 else InferenceMode.ZERO_SHOT

    all_aggregates: dict[str, dict] = {}
    t_total = time.time()

    for diff_name in args.difficulty:
        print(f"\n{'=' * 70}")
        print(f"  Backend    : {backend.value}")
        print(f"  Model      : {args.model}")
        print(f"  Difficulty : {diff_name.upper()}")
        print(f"  Few-shot   : {args.few_shot}")
        print(f"{'=' * 70}")

        diff_filter = DifficultyFilter.from_preset(
            diff_name,
            is_exclusive=True,
        )

        print("  Loading dataset ...")
        dataset = KakuroGrid(
            n_samples=args.n_samples + (args.few_shot or 0) + 100,
            difficulty=diff_filter,
            few_shot_strategy=args.few_shot_strategy,
            seed=args.seed,
            streaming=not args.no_streaming,
        )

        all_problems = list(dataset)
        print(f"  Loaded {len(all_problems):,} [{diff_name}] problems.")

        few_shot_examples: list[Problem] | None = None
        few_shot_ids: set[str] = set()

        if args.few_shot > 0:
            few_shot_examples = dataset.few_shot_examples(k=args.few_shot)
            few_shot_ids = {ex.problem_id for ex in few_shot_examples}
            print(
                f"  Reserving {len(few_shot_examples)} few-shot examples "
                f"excluded from test set."
            )

        test_pool = [
            p for p in all_problems
            if p.problem_id not in few_shot_ids
        ]

        if len(test_pool) > args.n_samples:
            rng = np.random.default_rng(args.seed)
            indices = rng.choice(len(test_pool), size=args.n_samples, replace=False)
            test_problems = [test_pool[int(i)] for i in sorted(indices)]
        else:
            test_problems = test_pool

        print(f"  Test set   : {len(test_problems):,} problems")

        t0 = time.time()

        result = evaluate_difficulty(
            backend=backend,
            model=model,
            tokenizer=tokenizer,
            client=client,
            model_name=args.model,
            problems=test_problems,
            difficulty_name=diff_name,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            mode=mode,
            few_shot_examples=few_shot_examples,
            temperature=args.temperature,
            openai_concurrency=args.openai_concurrency,
            openai_retries=args.openai_retries,
            openai_retry_sleep=args.openai_retry_sleep,
            n_success_samples=args.n_success_samples,
            n_failure_samples=args.n_failure_samples,
        )

        elapsed = time.time() - t0
        agg = result["aggregate"]
        all_aggregates[diff_name] = agg

        print(f"\n  Results -- {diff_name} ({shot_tag})")
        print(f"  {'Evaluated':<20}: {agg['n_total']:>10,}")
        print(
            f"  {'Parse failures':<20}: "
            f"{agg['n_parse_fail']:>10,} ({agg['pct_parse_fail']:.1f}%)"
        )
        print(f"  {'Cell accuracy':<20}: {agg['cell_acc']:>10.4f}")
        print(f"  {'Board accuracy':<20}: {agg['board_acc']:>10.4f}")
        print(f"  {'Row accuracy':<20}: {agg['row_acc']:>10.4f}")
        print(f"  {'Col accuracy':<20}: {agg['col_acc']:>10.4f}")
        print(f"  {'Wall time':<20}: {elapsed:>10.1f}s")

        print(f"\n  Saving to {output_dir}/")
        out_prefix = output_dir / f"{backend.value}_{model_tag}_{diff_name}_{shot_tag}"
        save_results(result, out_prefix, args)

    combined_path = output_dir / f"{backend.value}_{model_tag}_combined_{shot_tag}.json"

    with open(combined_path, "w") as f:
        json.dump(
            {
                "config": vars(args),
                "backend": backend.value,
                "model_tag": model_tag,
                "shot_tag": shot_tag,
                "total_time_s": round(time.time() - t_total, 1),
                "results": all_aggregates,
            },
            f,
            indent=2,
        )

    print(f"\n{'=' * 70}")
    print(f"  Combined summary -> {combined_path}")
    print(f"  Total wall time  : {time.time() - t_total:.1f}s")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
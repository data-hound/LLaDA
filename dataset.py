"""
Dataset loading and difficulty filtering for the NineGrid benchmark.

  kaggle datasets download beta3logic/ninegrid-sudoku-reasoning-benchmark-for-llms
  unzip ninegrid-sudoku-reasoning-benchmark-for-llms.zip
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .eval_types import Difficulty, Problem


# ---------------------------------------------------------------------------
# DifficultyFilter
# ---------------------------------------------------------------------------

class DifficultyFilter:
    """
    Restrict problems by difficulty. Use DifficultyFilter.from_preset() for
    the easy / medium / hard tiers, or pass raw thresholds directly.

    Presets
    -------
    easy   — missing_cells [17, 35], initial_resolution_rate [0.5, 1.0]
              Most cells are given; empties solvable by singles alone.
    medium — missing_cells [36, 49], initial_resolution_rate [0.2, 0.5]
              Balanced — requires some deduction beyond singles.
    hard   — missing_cells [50, 64], initial_resolution_rate [0.0, 0.2]
              Few givens; backtracking / advanced techniques needed.

    All thresholds are inclusive on both ends. None = unconstrained.
    """

    _PRESETS: dict[Difficulty, dict] = {
        Difficulty.EASY:   {"missing_cells": (17, 35),  "initial_resolution_rate": (0.5, 1.0)},
        Difficulty.MEDIUM: {"missing_cells": (36, 49),  "initial_resolution_rate": (0.2, 0.5)},
        Difficulty.HARD:   {"missing_cells": (50, 64),  "initial_resolution_rate": (0.0, 0.2)},
    }

    def __init__(
        self,
        missing_cells: tuple[int, int] | None = None,
        given_ratio: tuple[float, float] | None = None,
        naked_singles_count: tuple[int, int] | None = None,
        hidden_singles_count: tuple[int, int] | None = None,
        initial_resolution_rate: tuple[float, float] | None = None,
        preset: Difficulty | None = None,
    ):
        if preset is not None:
            p = self._PRESETS[preset]
            missing_cells            = missing_cells            or p.get("missing_cells")
            initial_resolution_rate  = initial_resolution_rate  or p.get("initial_resolution_rate")

        self.missing_cells           = missing_cells
        self.given_ratio             = given_ratio
        self.naked_singles_count     = naked_singles_count
        self.hidden_singles_count    = hidden_singles_count
        self.initial_resolution_rate = initial_resolution_rate
        self.preset                  = preset

    @classmethod
    def from_preset(cls, difficulty: Difficulty | str) -> "DifficultyFilter":
        """
        DifficultyFilter.from_preset(Difficulty.EASY)
        DifficultyFilter.from_preset("hard")
        """
        return cls(preset=Difficulty(difficulty))

    def matches(self, meta: dict) -> bool:
        checks = [
            ("missing_cells",           self.missing_cells),
            ("given_ratio",             self.given_ratio),
            ("naked_singles_count",     self.naked_singles_count),
            ("hidden_singles_count",    self.hidden_singles_count),
            ("initial_resolution_rate", self.initial_resolution_rate),
        ]
        for key, rng in checks:
            if rng is not None and key in meta:
                lo, hi = rng
                if not (lo <= meta[key] <= hi):
                    return False
        return True

    def __repr__(self) -> str:
        if self.preset:
            return f"DifficultyFilter(preset={self.preset.value!r})"
        parts = []
        for attr in ["missing_cells", "given_ratio", "naked_singles_count",
                     "hidden_singles_count", "initial_resolution_rate"]:
            v = getattr(self, attr)
            if v is not None:
                parts.append(f"{attr}={v}")
        return f"DifficultyFilter({', '.join(parts)})"


# ---------------------------------------------------------------------------
# NineGrid dataset
# ---------------------------------------------------------------------------

class NineGrid:
    """
    Loads the NineGrid Sudoku Reasoning Benchmark for LLMs from Kaggle.

    Parameters
    ----------
    parquet_path     : path to .parquet (preferred) or .csv download
    n_samples        : cap on number of problems to load
    difficulty       : DifficultyFilter to apply during loading
    few_shot_strategy: "easiest" | "hardest" | "random"
    seed             : RNG seed for random few-shot strategy
    """

    name = "NineGrid"

    _META_COLS = [
        "missing_cells", "given_cells", "given_ratio",
        "row_given_counts", "col_given_counts",
        "naked_singles_count", "hidden_singles_count",
        "initial_resolution_rate",
    ]

    def __init__(
        self,
        parquet_path: str,
        n_samples: int | None = None,
        difficulty: DifficultyFilter | None = None,
        few_shot_strategy: str = "easiest",
        seed: int = 42,
    ):
        self._rng = np.random.default_rng(seed)
        self._few_shot_strategy = few_shot_strategy
        self._problems = self._load(parquet_path, n_samples, difficulty)
        if not self._problems:
            raise ValueError("No problems loaded — check the path and difficulty filter.")

    # ---- ProblemSet protocol ------------------------------------------------

    def __len__(self) -> int:
        return len(self._problems)

    def __iter__(self):
        return iter(self._problems)

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
            indices = self._rng.choice(len(self._problems), size=k, replace=False)
            return [self._problems[i] for i in indices]

    # ---- inspection helpers -------------------------------------------------

    def filter(self, difficulty: DifficultyFilter) -> "NineGrid":
        """Return a new NineGrid with only problems matching the filter."""
        filtered = [p for p in self._problems if difficulty.matches(p.metadata)]
        clone = object.__new__(NineGrid)
        clone._rng = self._rng
        clone._few_shot_strategy = self._few_shot_strategy
        clone._problems = filtered
        return clone

    def difficulty_summary(self) -> dict:
        """Descriptive stats over difficulty metadata fields."""
        stats = {}
        for col in ["missing_cells", "given_ratio", "naked_singles_count",
                    "hidden_singles_count", "initial_resolution_rate"]:
            vals = [p.metadata[col] for p in self._problems if col in p.metadata]
            if vals:
                arr = np.array(vals, dtype=float)
                stats[col] = {
                    "min":    float(arr.min()),
                    "max":    float(arr.max()),
                    "mean":   float(arr.mean()),
                    "median": float(np.median(arr)),
                }
        return stats

    def count_by_difficulty(self) -> dict[str, int]:
        """How many problems fall into each preset difficulty tier."""
        counts = {}
        for d in Difficulty:
            f = DifficultyFilter.from_preset(d)
            counts[d.value] = sum(1 for p in self._problems if f.matches(p.metadata))
        counts["total"] = len(self._problems)
        return counts

    def info(self) -> None:
        """Print a human-readable summary of the loaded dataset."""
        counts  = self.count_by_difficulty()
        summary = self.difficulty_summary()
        print("NineGrid Dataset")
        print(f"  Total problems : {len(self._problems):,}")
        print()
        print("  By difficulty tier:")
        for d in Difficulty:
            print(f"    {d.value:<8} : {counts[d.value]:>9,} problems")
        print()
        print("  Difficulty field ranges (across loaded problems):")
        for col, s in summary.items():
            print(f"    {col:<28} min={s['min']:.2f}  max={s['max']:.2f}  mean={s['mean']:.2f}")

    # ---- loading ------------------------------------------------------------

    def _load(
        self,
        path: str,
        n_samples: int | None,
        difficulty: DifficultyFilter | None,
    ) -> list[Problem]:
        import ast
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError("Install `pandas` and `pyarrow` to load NineGrid.") from e

        df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)

        problems: list[Problem] = []
        for idx, row in df.iterrows():
            puzzle_grid   = self._coerce_grid(row["puzzle_grid"])
            solution_grid = self._coerce_grid(row["solution_grid"])
            grid = [[v if v != 0 else None for v in r] for r in puzzle_grid]

            meta: dict[str, Any] = {}
            for col in self._META_COLS:
                if col in row.index:
                    val = row[col]
                    if isinstance(val, str):
                        try:
                            val = ast.literal_eval(val)
                        except Exception:
                            pass
                    meta[col] = val

            if difficulty is not None and not difficulty.matches(meta):
                continue

            problems.append(Problem(
                problem_id=str(row.get("id", idx)),
                grid=grid,
                solution=solution_grid,
                metadata=meta,
            ))

            if n_samples is not None and len(problems) >= n_samples:
                break

        return problems

    @staticmethod
    def _coerce_grid(value: Any) -> list[list[int]]:
        import ast
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            return ast.literal_eval(value)
        raise ValueError(f"Cannot parse grid from type {type(value)}: {value!r}")

class FourGridDifficultyFilter(DifficultyFilter):
    _PRESETS: dict[Difficulty, dict] = {
        Difficulty.EASY:   {"difficulty": 'easy'},
        Difficulty.MEDIUM: {"difficuly": 'medium'},
        Difficulty.HARD:   {"difficuly": 'hard'},
    }

    def __init__(
        self,
        missing_cells: tuple[int, int] | None = None,
        given_ratio: tuple[float, float] | None = None,
        naked_singles_count: tuple[int, int] | None = None,
        hidden_singles_count: tuple[int, int] | None = None,
        initial_resolution_rate: tuple[float, float] | None = None,
        difficulty: str | None = None,
        preset: Difficulty | None = None,
    ):
        if preset is not None:
            p = self._PRESETS[preset]
            difficulty               = difficulty            or p.get("difficulty")

        self.difficulty              = difficulty  
        self.missing_cells           = missing_cells
        self.given_ratio             = given_ratio
        self.naked_singles_count     = naked_singles_count
        self.hidden_singles_count    = hidden_singles_count
        self.initial_resolution_rate = initial_resolution_rate
        self.preset                  = preset
    
    def matches(self, meta: dict) -> bool:
        checks = [
            ("difficulty",              self.difficulty),
            ("missing_cells",           self.missing_cells),
            ("given_ratio",             self.given_ratio),
            ("naked_singles_count",     self.naked_singles_count),
            ("hidden_singles_count",    self.hidden_singles_count),
            ("initial_resolution_rate", self.initial_resolution_rate),
        ]
        for key, rng in checks:
            if rng is not None and key in meta:
                lo, hi = rng
                if not (lo <= meta[key] <= hi):
                    return False
        return True

    def __repr__(self) -> str:
        if self.preset:
            return f"DifficultyFilter(preset={self.preset.value!r})"
        parts = []
        for attr in ["difficulty", "missing_cells", "given_ratio", "naked_singles_count",
                     "hidden_singles_count", "initial_resolution_rate"]:
            v = getattr(self, attr)
            if v is not None:
                parts.append(f"{attr}={v}")
        return f"DifficultyFilter({', '.join(parts)})"

class FourGrid(NineGrid):

    name="FourGrid"
    _META_COLS = [
        "board_id",
        "root_hash",
        "solution_id",
        "size",
        "difficulty",
        "missing_cells",
    ]
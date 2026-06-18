#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
export_next_v2.py

Finalni izvoz 3 kombinacija (3 modela x 1) u next_v2.txt — PUNA optimizacija:
  A — Pipeline (freq + pair + structure)
  B — Advanced (Optuna tezine + MC 12000 + MCTS 5000 + recent + nextgen)
  C — Enhanced (obrasci iz backtest detalja ako postoji)

Podrazumevano: najbolja next predikcija, ne brzi rezim. Za test: --fast.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Sequence, Tuple

from loto7_logic_predictor import DEFAULT_CSV, format_ticket, load_draws
from loto7_pipeline import generate_candidates

DEFAULT_TXT = "next_v2.txt"
NUM_PER_MODEL = 1
SEED = 39
# Isti pool kao loto7_logic_predictor --pool-size 24
POOL_SIZE = 24
HIT_PATTERN_CSV = "loto7_backtest_detail.csv"

# Pun režim = podrazumevane vrednosti iz loto7_advanced_optimizer (ne prepisuj env)
_FULL_ENV_KEYS = (
    "LOTO7_DISABLE_OPTIMIZE",
    "LOTO7_MONTE_CARLO",
    "LOTO7_MCTS_ITERATIONS",
    "LOTO7_DISABLE_NEXTGEN",
    "LOTO7_SKIP_RECENT",
)


def _apply_mode(fast: bool) -> None:
    """Ukloni stare fast override-e; opciono ukljuci --fast."""
    for key in _FULL_ENV_KEYS:
        os.environ.pop(key, None)
    if fast:
        os.environ["LOTO7_DISABLE_OPTIMIZE"] = "1"
        os.environ["LOTO7_MONTE_CARLO"] = "100"
        os.environ["LOTO7_MCTS_ITERATIONS"] = "0"
        os.environ["LOTO7_DISABLE_NEXTGEN"] = "1"
        os.environ["LOTO7_SKIP_RECENT"] = "1"


def _fmt_combo(combo: Sequence[int]) -> str:
    return " ".join(f"{n:02d}" for n in sorted(combo))


def _best_ticket(tickets, n: int = 1) -> List[Tuple[int, ...]]:
    return [tuple(sorted(t.ticket)) for t in tickets[:n]]


def collect_all_combos(csv_path: str, num_per_model: int = NUM_PER_MODEL, fast: bool = False):
    _apply_mode(fast)

    draws = load_draws(csv_path)
    if len(draws) < 60:
        raise ValueError(f"Premalo izvlacenja: {len(draws)}")

    from loto7_advanced_optimizer import advanced_predict
    from loto7_enhanced_predictor import enhanced_predict

    mode = "BRZI" if fast else "PUNA OPTIMIZACIJA"
    print(f"[REZIM] {mode} | pool={POOL_SIZE} | seed={SEED}", flush=True)

    print("[A] Pipeline (logika)...", flush=True)
    model_a = generate_candidates(draws, purchase_count=num_per_model, pool_size=POOL_SIZE)

    print("[B] Napredni optimizer (Optuna + MC + MCTS)...", flush=True)
    model_b = _best_ticket(
        advanced_predict(
            draws,
            num_tickets=num_per_model,
            pool_size=POOL_SIZE,
            hit_pattern_csv=HIT_PATTERN_CSV,
            optimize=not fast,
            monte_carlo_iterations=150 if fast else None,
            mcts_iterations=0 if fast else None,
        ),
        num_per_model,
    )

    print("[C] Poboljsani prediktor...", flush=True)
    model_c = _best_ticket(
        enhanced_predict(
            draws,
            num_tickets=num_per_model,
            pool_size=POOL_SIZE,
            hit_pattern_csv=HIT_PATTERN_CSV,
        ),
        num_per_model,
    )
    return draws, model_a, model_b, model_c, fast


def build_text(draws, model_a, model_b, model_c, csv_path: str, fast: bool = False) -> str:
    latest = draws[-1]
    next_kolo = (latest.draw_no or len(draws)) + 1
    a = model_a[0] if model_a else tuple()
    b = model_b[0] if model_b else tuple()
    c = model_c[0] if model_c else tuple()
    rezim = "brzi test" if fast else "puna optimizacija (Optuna+MC12000+MCTS5000)"
    lines = [
        "NEW_LOTO7_v2 — next predikcija (3 kombinacije = 3×1)",
        f"CSV: {csv_path}",
        f"Poslednje kolo: {latest.draw_no} | Sledece: {next_kolo}",
        f"Seed: {SEED} | Rezim: {rezim}",
        "",
        f"Model A — Pipeline: {_fmt_combo(a)}",
        f"Model B — Advanced: {_fmt_combo(b)}",
        f"Model C — Enhanced: {_fmt_combo(c)}",
        "",
        "FINALNE 3 kombinacije (A, B, C):",
        f"  1: {_fmt_combo(a)}",
        f"  2: {_fmt_combo(b)}",
        f"  3: {_fmt_combo(c)}",
        "",
    ]
    return "\n".join(lines)


def export_next_v2(csv_path: str = DEFAULT_CSV, output_path: str = DEFAULT_TXT, fast: bool = False) -> str:
    draws, a, b, c, fast = collect_all_combos(csv_path, fast=fast)
    text = build_text(draws, a, b, c, csv_path, fast=fast)
    out = Path(output_path)
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"Sacuvano: {out.resolve()}")
    return str(out.resolve())


def main() -> int:
    p = argparse.ArgumentParser(description="Izvoz 3 kombinacije u next_v2.txt (podrazumevano: puna optimizacija)")
    p.add_argument("--csv", default=DEFAULT_CSV)
    p.add_argument("--output", default=DEFAULT_TXT)
    p.add_argument("--fast", action="store_true", help="Samo za brzi test (~2 min), NE za finalnu predikciju")
    args = p.parse_args()
    export_next_v2(args.csv, args.output, fast=args.fast)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


"""
export_next_v2.py

[A] Pipeline (logika)...
[B] Napredni optimizer...
[C] Poboljsani prediktor...

next predikcija (3 kombinacije = 3x1)
CSV: /data/loto7hh_4634_k48.csv
Poslednje kolo: 4634 | Sledece: 4635
Seed: 39

Model A — Pipeline: 01 x 14 y 24 z 38
Model B — Advanced: 01 x 08 y 30 z 38
Model C — Enhanced: 01 x 08 y 24 z 38

FINALNE 3 kombinacije (A, B, C):
  1: 01 x 14 y 24 z 38
  2: 01 x 08 y 30 z 38
  3: 01 x 08 y 24 z 38

Sacuvano: /next_v2.txt
"""

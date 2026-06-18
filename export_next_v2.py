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

Model A — Pipeline: 01 08 14 16 24 34 38
Model B — Advanced: 01 03 08 19 30 31 38
Model C — Enhanced: 01 03 08 16 24 31 38

FINALNE 3 kombinacije (A, B, C):
  1: 01 08 14 16 24 34 38
  2: 01 03 08 19 30 31 38
  3: 01 03 08 16 24 31 38

Sacuvano: /next_v2.txt
"""

"""
cd /Users/4c/Desktop/GHQ/STATISTIKA/NEW_LOTO7_v2-main
PYTHONUNBUFFERED=1 /Users/4c/tesla_env/bin/python export_next_v2.py --fast

[REZIM] BRZI | pool=24 | seed=39
[A] Pipeline (logika)...
[B] Napredni optimizer (Optuna + MC + MCTS)...



--fast na celom CSV (4634 kola) nije 2 min — u kodu i dalje ide 735k kombinacija za B i za C. Isključeni su samo Optuna, recent, MCTS (veći).

Deo	--fast
A
~1 min
B
~1–4 h
C
~1–4 h
Ukupno
~2–8 h (M1)
Ako CPU ~100% — radi, samo čekaj. Log stoji na [B] dok B ne završi.

"""








"""
cd "/Users/4c/Desktop/GHQ/STATISTIKA/NEW_LOTO7_v2-main"

# 1. ceo CSV
"/Users/4c/tesla_env/bin/python" loto7_logic_predictor.py --backtest --max-backtest-draws 0

# 2. final
PYTHONUNBUFFERED=1 "/Users/4c/tesla_env/bin/python" export_next_v2.py
"""



"""
NEW_LOTO7_v2 — analiza projekta
Adaptacija japanskog NEW_LOTO7 pipeline-a za Srpski Loto 7/39: 4634 kola, CSV Num1..Num7, seed 39, bez scrapinga i bez logike nagrada.

Arhitektura (3 modela → next_v2.txt)
Model	Fajl	Suština
A — Pipeline
loto7_pipeline.py
Stdlib: frekvencija, recency, gap, parovi, struktura (nizak/srednji/visok, parno)
B — Advanced
loto7_advanced_optimizer.py
Teški optimizer: exhaustive pool + MC + MCTS + MemoryBank + Optuna + meta-regresor (RF/CatBoost/XGBoost)
C — Enhanced
loto7_enhanced_predictor.py
Vremensko opadanje, par/trojka, ponovno korišćenje visokih pogodaka iz loto7_backtest_detail.csv
Ulaz: loto7_logic_predictor.py (predikcija + advanced backtest)
Finalni izlaz: export_next_v2.py → next_v2.txt
Podaci: data/loto7hh_4634_k48.csv

Metode i tehnike
Statistika / heuristika

Walk-forward validacija (bez future leak-a)
Frekvencije, parovi, trojke sa vremenskim opadanjem
Strukturni filteri: suma, parno/neparno, opsezi 1–13 / 14–26 / 27–39, gap, serije
Mašinsko učenje (samo regresori, kako si tražio)

RF + CatBoost + XGBoost u train_meta_regressor() — predviđaju očekivani broj pogodaka (rang, ne verovatnoća dobitka)
Optuna (25 trial-a) za težine scoring funkcije
KMeans klasteri visokih pogodaka (opciono, sklearn)
Pretraga kandidata

Exhaustive kombinacije iz pool-a (npr. C(24,7) ≈ 735k)
Monte Carlo (podrazumevano 12 000 iteracija)
MCTS (podrazumevano 5 000)
Recent prozori 240/120/60 kola
MemoryBank za 4+ / 5+ pogodake iz prošlih predikcija
Inženjering

Nastavljivi backtest (resume_state.json)
Seed 39 fiksiran
CSV kolone na srpskom (kolo, predikcija1, pogodaka, …)
Naučna / statistička realnost
Loto 7/39 je nezavisno izvlačenje — prošla kola ne menjaju verovatnoću sledećeg. Modeli traže strukturisane kandidate koji u istoriji imaju bolji profil pogodaka u walk-forward testu; to nije dokaz prediktivne moći na budućem kolu.

Score = rang kandidata, ne P(dobitak).

Prednosti
Više nezavisnih pristupa (A/B/C), manje zavisnosti od jednog algoritma
Walk-forward dizajn — ispravan za ovakav zadatak
Bogat feature engineering (parovi, struktura, recency)
Puna optimizacija u export_next_v2.py (bez fast režima po defaultu)
Prilagođen srpskom formatu 1–39, bez japanskog CSV šuma
Nastavak prekinutog backtesta
Mane
Ekstremno spor pun režim (sati/dani na celom CSV)
logic_predictor pre backtesta radi pun advanced_predict — gubi se vreme pre nego što backtest krene
loto7_backtest_detail.csv ne postoji — Model C i MemoryBank za B rade bez punog istorijskog sloja
Pipeline backtest nije na celom CSV — trenutno ~70 kola (do draw 170), ne 4534
next_v2.txt na disku = stara 3×2 verzija; kod sada cilja 3×1
Nextgen (diffusion/PPO) zavisi od fajlova koji verovatno nisu u projektu — tiho se preskače
Backtest metrike ≈ slučajan tiket (vidi ispod) — nema dokazane prednosti nad randomom
Trenutno stanje fajlova
Fajl	Status
loto7_backtest_result.csv
Pipeline WF, 140 tiketa, draw 101–170
loto7_backtest_summary.csv
max pogodaka 3, prosek ~1.19
loto7_backtest_detail.csv
nema (advanced backtest nije završen)
next_v2.txt
6 kombinacija (stari export), kolo 4635
loto7_latest_prediction.csv
1 kombinacija (A pipeline)
Kratka analiza rezultata
Poslednje kolo 4634: 01 03 08 09 30 31 38

Predikcije u next_v2.txt (stari run):

Tiket	Pogodaka vs 4634	Napomena
B1, C1
7/7
praktično kopija poslednjeg kola
B2, C2
4/7
jaka recency
A1
3/7
raznovrsniji
A2
2/7
najmanje vezan za 4634
B i C su jako recency-biased — dobro „pogađaju“ upravo poslednje kolo u retrospektivi, ali to ne znači predikciju za 4635.

Pipeline backtest (70 kola, 2 tiketa/kolo):

Pogodaka	Model	Slučajan tiket
0
23.6%
21.9%
1
42.1%
41.2
"""

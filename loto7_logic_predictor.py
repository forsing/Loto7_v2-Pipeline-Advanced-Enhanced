#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
loto7_logic_predictor.py

v2 — ulazna tacka za predikciju i walk-forward backtest (Srpski Loto 7/39).

Koristi Advanced Optimizer direktno. Podaci: samo CSV sa Num1..Num7.
Seed: 39. Podrazumevano 1 kombinacija.

Napomena: score je rang, ne verovatnoca dobitka.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

SEED = 39
DEFAULT_CSV = str(Path(__file__).resolve().parent.parent.parent / "data" / "loto7hh_4634_k48.csv")
NUM_COLS = [f"Num{i}" for i in range(1, 8)]

DEFAULT_OUTPUT_CSV = "loto7_predictions.csv"
DEFAULT_LATEST_TXT = "latest_loto7_prediction.txt"

DEFAULT_BACKTEST_SUMMARY_CSV = "loto7_backtest_summary.csv"
DEFAULT_BACKTEST_DETAIL_CSV = "loto7_backtest_detail.csv"
DEFAULT_BACKTEST_REPORT_TXT = "loto7_backtest_report.txt"

NUM_MIN = 1
NUM_MAX = 39
PICK_SIZE = 7

DEFAULT_TICKETS = 1
DEFAULT_SAVE_COUNT = 1
DEFAULT_BACKTEST_POOL_CAP = 16


@dataclass(frozen=True)
class Draw:
    date: str
    main: Tuple[int, ...]
    bonus: Tuple[int, ...]
    draw_no: Optional[int] = None


@dataclass
class TicketScore:
    ticket: Tuple[int, ...]
    score: float
    detail: Dict[str, float]
    strategy: str = "CORE"


def count_main_matches(ticket: Sequence[int], actual_main: Sequence[int]) -> int:
    """Broj pogodaka u kombinaciji (0..7)."""
    return len(set(ticket) & set(actual_main))


def validate_main_numbers(nums: Sequence[int]) -> Tuple[int, ...]:
    if len(nums) != PICK_SIZE:
        raise ValueError(f"Potrebno je 7 brojeva: {nums}")
    if len(set(nums)) != PICK_SIZE:
        raise ValueError(f"Duplikati u kombinaciji: {nums}")
    for n in nums:
        if not (NUM_MIN <= n <= NUM_MAX):
            raise ValueError(f"Broj van opsega: {n}")
    return tuple(sorted(nums))


def load_draws(source: str = DEFAULT_CSV) -> List[Draw]:
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"CSV nije pronadjen: {source}")
    draws: List[Draw] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            try:
                main = validate_main_numbers([int(row[c]) for c in NUM_COLS])
            except (KeyError, ValueError):
                continue
            draws.append(Draw(date=str(idx), main=main, bonus=tuple(), draw_no=idx))
    draws.sort(key=lambda d: d.draw_no or 0)
    return draws


def format_ticket(ticket: Sequence[int], zero_pad: bool = True) -> str:
    if zero_pad:
        return ", ".join(f"{n:02d}" for n in sorted(ticket))
    return ", ".join(str(n) for n in sorted(ticket))


def score_normalized_values(ranked: Sequence[TicketScore]) -> List[float]:
    if not ranked:
        return []
    scores = [x.score for x in ranked]
    max_score = max(scores)
    min_score = min(scores)
    if max_score == min_score:
        return [1.000 for _ in ranked]
    return [round((s - min_score) / (max_score - min_score), 3) for s in scores]


def count_numbers(draws: Sequence[Draw], window: int) -> Counter:
    c: Counter = Counter()
    for d in draws[-window:] if window > 0 else draws:
        c.update(d.main)
    return c


def count_combinations(draws: Sequence[Draw], window: int, k: int) -> Counter:
    import itertools

    c: Counter = Counter()
    for d in draws[-window:] if window > 0 else draws:
        for comb in itertools.combinations(sorted(d.main), k):
            c[comb] += 1
    return c


def print_recent_summary(draws: Sequence[Draw]) -> None:
    latest = draws[-1]
    print("=== Poslednje izvlacenje ===")
    print(f"Kolo: {latest.draw_no}")
    print(f"Kombinacija: {format_ticket(latest.main)}")
    print()

    print("=== Pojave u poslednjih 10 kola ===")
    c10 = count_numbers(draws, 10)
    for n, cnt in sorted(c10.items(), key=lambda x: (-x[1], x[0])):
        print(f"{n:02d}: {cnt}x")
    print()

    print("=== Najjaci parovi (poslednjih 20) TOP15 ===")
    p20 = count_combinations(draws, 20, 2)
    for pair, cnt in p20.most_common(15):
        print(f"{pair[0]:02d}-{pair[1]:02d}: {cnt}x")
    print()


def print_predictions(tickets: Sequence[TicketScore]) -> None:
    print("=== Napredni optimizer — sledeca predikcija ===")
    scores = score_normalized_values(tickets)
    for i, t in enumerate(tickets, start=1):
        d = t.detail
        score = scores[i - 1] if i - 1 < len(scores) else 0.0
        print(
            f"{i:02d}. {format_ticket(t.ticket)}"
            f" | score={score:.3f}"
            f" | strategija={t.strategy}"
            f" | raw={t.score:.3f}"
            f" | suma={int(d.get('sum', 0))}"
            f" | parno={int(d.get('odd', 0))}"
        )
    print()


def prediction_csv_header(save_count: int = DEFAULT_SAVE_COUNT) -> List[str]:
    header = ["kolo"]
    for i in range(1, save_count + 1):
        header.extend([f"predikcija{i}", f"score{i}", f"strategija{i}"])
    return header


def _csv_kolo(row: Dict[str, str]) -> str:
    return row.get("kolo") or row.get("抽せん日", "")


def _csv_pred(row: Dict[str, str], i: int) -> str:
    return row.get(f"predikcija{i}") or row.get(f"予測{i}", "")


def _csv_score(row: Dict[str, str], i: int) -> str:
    return row.get(f"score{i}") or row.get(f"スコア正規化値{i}") or row.get(f"信頼度{i}", "")


def _csv_strategija(row: Dict[str, str], i: int) -> str:
    return row.get(f"strategija{i}") or row.get(f"戦略{i}", "")


def save_predictions_csv(output_path: str, target_date: str, ranked: Sequence[TicketScore], save_count: int) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = prediction_csv_header(save_count)

    scores = score_normalized_values(ranked[:save_count])
    new_row = {key: "" for key in header}
    new_row["kolo"] = target_date
    for i, ticket in enumerate(ranked[:save_count], start=1):
        new_row[f"predikcija{i}"] = format_ticket(ticket.ticket, zero_pad=False)
        new_row[f"score{i}"] = f"{scores[i - 1]:.3f}".rstrip("0").rstrip(".")
        new_row[f"strategija{i}"] = ticket.strategy

    rows: List[Dict[str, str]] = []
    if path.exists():
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            for old in csv.DictReader(f):
                row = {key: "" for key in header}
                row["kolo"] = _csv_kolo(old)
                for i in range(1, save_count + 1):
                    row[f"predikcija{i}"] = _csv_pred(old, i)
                    row[f"score{i}"] = _csv_score(old, i)
                    row[f"strategija{i}"] = _csv_strategija(old, i)
                rows.append(row)

    rows = [row for row in rows if _csv_kolo(row) != target_date]
    rows.append(new_row)
    rows.sort(key=lambda r: _csv_kolo(r))

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def write_compat_report(summary: Dict[str, object], report_path: str) -> None:
    out = Path(report_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    def _s(*keys: str) -> object:
        for k in keys:
            if k in summary and summary[k] not in ("", None):
                return summary[k]
        return ""

    lines = [
        "Loto7 Advanced Optimizer — izvestaj backtesta",
        "=============================================",
        "",
        "Osnovni uslovi",
        "----------------------------",
        f"Broj validacija: {_s('broj_validacija', '検証回数')}",
        f"Pocetna obuka: {_s('pocetna_obuka', '初期学習回数')}",
        f"Pocetak validacije (kolo): {_s('pocetak_validacije', '検証開始回相当')}",
        f"Kombinacija po kolu: {_s('komb_po_kolu', '1回あたり口数')}",
        f"Velicina pool-a: {_s('velicina_pool', '実効バックテスト候補プール', '候補プール')}",
        "",
        "Pogoci",
        "----------------------------",
        f"Prosecno pogodaka (1. komb): {_s('prosek_pog_prva', '1口目平均一致数')}",
        f"Prosecno max pogodaka: {_s('prosek_pog_max', '全口ベスト平均一致数')}",
        f"1. komb 3+ stopa: {_s('stopa_prva_3plus', '1口目_3個以上率')}",
        f"1. komb 4+ stopa: {_s('stopa_prva_4plus', '1口目_4個以上率')}",
        "",
        "Napomena: samo broj pogodaka, bez nagrada/novca.",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="NEW_LOTO7_v2 — predikcija i backtest (Srpski Loto 7/39)")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Putanja do CSV (Num1..Num7)")
    parser.add_argument("--tickets", type=int, default=DEFAULT_TICKETS, help="Broj kombinacija. Podrazumevano: 1")
    parser.add_argument("--pool-size", type=int, default=24, help="Velicina pool-a kandidata (vece = sporije). Podrazumevano: 24")
    parser.add_argument("--backtest", action="store_true", help="Pokreni i backtest")
    parser.add_argument("--min-train", type=int, default=100, help="Pocetna obuka za backtest (1 = od 2. kola)")
    parser.add_argument("--backtest-pool-size", type=int, default=16, help="Pool u backtestu. Podrazumevano: 16")
    parser.add_argument("--backtest-pool-cap", type=int, default=DEFAULT_BACKTEST_POOL_CAP, help="Gornja granica pool-a (timeout). Podrazumevano: 16")
    parser.add_argument("--max-backtest-draws", type=int, default=0, help="Samo poslednjih N kola (0 = sva)")
    parser.add_argument("--monte-carlo", type=int, default=None, help="Monte Carlo za predikciju (env ili podrazumevano)")
    parser.add_argument("--backtest-monte-carlo", type=int, default=None, help="Monte Carlo u backtestu (LOTO7_BACKTEST_MONTE_CARLO)")
    parser.add_argument("--disable-optimize", action="store_true", help="Iskljuci Optuna/Random Search; sacuvane ili podrazumevane tezine")
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV, help="Izlazni CSV predikcija")
    parser.add_argument("--latest-txt", default=DEFAULT_LATEST_TXT, help="Izlazni TXT najnovije predikcije")
    parser.add_argument("--save-count", type=int, default=DEFAULT_SAVE_COUNT, help="Koliko predikcija snimiti. Podrazumevano: 10")
    parser.add_argument("--no-save", action="store_true", help="Ne snimaj CSV/TXT")
    parser.add_argument("--backtest-summary-csv", default=DEFAULT_BACKTEST_SUMMARY_CSV)
    parser.add_argument("--backtest-detail-csv", default=DEFAULT_BACKTEST_DETAIL_CSV)
    parser.add_argument("--backtest-report-txt", default=DEFAULT_BACKTEST_REPORT_TXT)
    args = parser.parse_args(argv)

    if args.disable_optimize:
        import os
        os.environ["LOTO7_DISABLE_OPTIMIZE"] = "1"
    if args.monte_carlo is not None:
        import os
        os.environ["LOTO7_MONTE_CARLO"] = str(args.monte_carlo)
    if args.backtest_monte_carlo is not None:
        import os
        os.environ["LOTO7_BACKTEST_MONTE_CARLO"] = str(args.backtest_monte_carlo)

    from loto7_advanced_optimizer import advanced_predict, save_latest_txt, advanced_backtest

    draws = load_draws(args.csv)
    if not draws:
        print("Nije moguce ucitati CSV.", file=sys.stderr)
        return 1

    latest = draws[-1]
    target_date = str((latest.draw_no or len(draws)) + 1)

    print_recent_summary(draws)

    display_tickets = advanced_predict(
        draws,
        num_tickets=args.tickets,
        pool_size=args.pool_size,
        hit_pattern_csv=args.backtest_detail_csv,
        monte_carlo_iterations=args.monte_carlo,
        optimize=not args.disable_optimize,
    )
    print_predictions(display_tickets)

    if not args.no_save:
        save_predictions_csv(
            output_path=args.output_csv,
            target_date=target_date,
            ranked=display_tickets,
            save_count=args.save_count,
        )
        save_latest_txt(
            path=args.latest_txt,
            target_date=target_date,
            tickets=display_tickets[: args.save_count],
        )
        print(f"Sacuvano: {args.output_csv}")
        print(f"Sacuvano: {args.latest_txt}")
        print("Format: kolo, predikcija1, score1, strategija1 ...")
        print(f"Ciljno kolo: {target_date}")
        print()

    if args.backtest:
        max_backtest_draws = args.max_backtest_draws if args.max_backtest_draws > 0 else 0
        summary = advanced_backtest(
            draws,
            min_train=args.min_train,
            num_tickets=args.tickets,
            pool_size=min(args.backtest_pool_size, args.backtest_pool_cap) if args.backtest_pool_cap > 0 else args.backtest_pool_size,
            hit_pattern_csv=args.backtest_detail_csv,
            max_backtest_draws=max_backtest_draws,
            summary_csv=args.backtest_summary_csv,
            detail_csv=args.backtest_detail_csv,
        )
        write_compat_report(summary, args.backtest_report_txt)

        print("=== Advanced backtest — rezultat ===")
        for key, value in summary.items():
            print(f"{key}: {value}")
        print(f"Sacuvano: {args.backtest_summary_csv}")
        print(f"Sacuvano: {args.backtest_detail_csv}")
        print(f"Sacuvano: {args.backtest_report_txt}")
        print()

    print("Napomena: score je rang kandidata, ne verovatnoca dobitka.")

    from export_next_v2 import export_next_v2
    export_next_v2(args.csv, "next_v2.txt")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())



"""
cd ...

# 1. ceo CSV
loto7_logic_predictor.py --backtest --max-backtest-draws 0

# 2. final
export_next_v2.py
"""



"""
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

Mašinsko učenje (samo regresori)
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

Score = rang kandidata, ne P(dobitak).

Prednosti
Više nezavisnih pristupa (A/B/C), manje zavisnosti od jednog algoritma
Walk-forward dizajn — ispravan za ovakav zadatak
Bogat feature engineering (parovi, struktura, recency)
Puna optimizacija u export_next_v2.py (bez fast režima po defaultu)
Nastavak prekinutog backtesta

Mane
Ekstremno spor pun režim (sati/dani na celom CSV)
logic_predictor pre backtesta radi pun advanced_predict — gubi se vreme pre nego što backtest krene
loto7_backtest_detail.csv ne postoji — Model C i MemoryBank za B rade bez punog istorijskog sloja
Pipeline backtest nije na celom CSV — trenutno ~70 kola (do draw 170), ne 4534
next_v2.txt na disku = 3×2 3×1
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
6(3) kombinacija, sledece kolo 4635
loto7_latest_prediction.csv
1 kombinacija (A pipeline)
Kratka analiza rezultata
Poslednje kolo 4634: 01 03 08 09 30 31 38

Predikcije u next_v2.txt:

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

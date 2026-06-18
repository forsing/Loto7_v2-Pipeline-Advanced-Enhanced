#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
loto7_pipeline.py

v2 — lagani, nastavljivi pipeline za Srpski Loto 7/39.

Cilj:
  1. Ucitavanje samo CSV-a sa 7 kolona brojeva (Num1..Num7)
  2. Walk-forward backtest bez future leak-a
  3. Nastavak preko resume_state.json i CSV rezultata
  4. Podrazumevano 1 kombinacija za next predikciju

Samo standardna biblioteka. Podaci: loto7hh_4634_k48.csv. Seed: 39.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import itertools
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

SEED = 39
NUMBERS = tuple(range(1, 40))
DEFAULT_CSV = str(Path(__file__).resolve().parent.parent.parent / "data" / "loto7hh_4634_k48.csv")
NUM_COLS = [f"Num{i}" for i in range(1, 8)]


@dataclass(frozen=True)
class Draw:
    draw_no: int
    date: str
    main: Tuple[int, ...]
    bonus: Tuple[int, ...]


def parse_nums(text: object) -> Tuple[int, ...]:
    raw = str(text or "").replace(",", " ").split()
    nums = tuple(int(x) for x in raw if str(x).isdigit())
    return nums


def draw_no_int(text: object) -> Optional[int]:
    import re

    m = re.search(r"\d+", str(text or ""))
    return int(m.group(0)) if m else None


def load_draws(csv_path: str) -> List[Draw]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV nije pronadjen: {csv_path}")

    draws: List[Draw] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            try:
                main = tuple(sorted(int(row[c]) for c in NUM_COLS))
            except (KeyError, ValueError):
                continue
            if len(main) != 7 or len(set(main)) != 7:
                continue
            if any(n < 1 or n > 39 for n in main):
                continue
            draws.append(Draw(draw_no=idx, date=str(idx), main=main, bonus=tuple()))

    draws.sort(key=lambda d: d.draw_no)
    return draws


def save_json(path: str, payload: Dict[str, object]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def load_json(path: str) -> Dict[str, object]:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def append_csv(path: str, fieldnames: Sequence[str], rows: Iterable[Dict[str, object]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    exists = p.exists() and p.stat().st_size > 0
    with p.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_csv(path: str, fieldnames: Sequence[str], rows: Iterable[Dict[str, object]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def score_numbers(train: Sequence[Draw]) -> Dict[int, float]:
    """Balans score: frekvencija, recency, gap. Bez future podataka."""
    freq = {n: 0.0 for n in NUMBERS}
    last_seen = {n: -1 for n in NUMBERS}
    total = len(train)

    for idx, draw in enumerate(train):
        age = total - idx - 1
        weight = 0.985 ** age
        for n in draw.main:
            freq[n] += weight
            last_seen[n] = idx

    scores: Dict[int, float] = {}
    for n in NUMBERS:
        gap = total - last_seen[n] - 1 if last_seen[n] >= 0 else total
        scores[n] = freq[n] + min(gap, 30) * 0.018
    return scores


def affinity_scores(train: Sequence[Draw]) -> Dict[Tuple[int, int], float]:
    aff: Dict[Tuple[int, int], float] = {}
    total = len(train)
    for idx, draw in enumerate(train):
        age = total - idx - 1
        weight = 0.99 ** age
        for a, b in itertools.combinations(draw.main, 2):
            key = (min(a, b), max(a, b))
            aff[key] = aff.get(key, 0.0) + weight
    return aff


def combo_score(combo: Sequence[int], num_score: Dict[int, float], aff: Dict[Tuple[int, int], float]) -> float:
    score = sum(num_score[n] for n in combo)
    score += sum(aff.get((min(a, b), max(a, b)), 0.0) * 0.07 for a, b in itertools.combinations(combo, 2))

    odd = sum(1 for n in combo if n % 2)
    low = sum(1 for n in combo if n <= 20)
    total = sum(combo)

    if odd in (3, 4):
        score += 0.35
    else:
        score -= 0.30
    if low in (3, 4):
        score += 0.25
    else:
        score -= 0.25
    if 100 <= total <= 185:
        score += 0.30
    else:
        score -= 0.40

    consecutive_pairs = sum(1 for a, b in zip(combo, combo[1:]) if b == a + 1)
    if consecutive_pairs <= 1:
        score += 0.15
    else:
        score -= consecutive_pairs * 0.25
    return score


def generate_candidates(train: Sequence[Draw], purchase_count: int = 1, pool_size: int = 18) -> List[Tuple[int, ...]]:
    if not train:
        raise ValueError("obuka je prazna")
    num_score = score_numbers(train)
    aff = affinity_scores(train)
    pool = [n for n, _ in sorted(num_score.items(), key=lambda kv: kv[1], reverse=True)[:pool_size]]
    pool = sorted(pool)

    scored: List[Tuple[float, Tuple[int, ...]]] = []
    for combo in itertools.combinations(pool, 7):
        scored.append((combo_score(combo, num_score, aff), tuple(combo)))
    scored.sort(reverse=True, key=lambda x: x[0])

    selected: List[Tuple[int, ...]] = []
    for _, combo in scored:
        if all(len(set(combo) & set(prev)) <= 5 for prev in selected):
            selected.append(combo)
        if len(selected) >= purchase_count:
            break

    if len(selected) < purchase_count:
        for _, combo in scored:
            if combo not in selected:
                selected.append(combo)
            if len(selected) >= purchase_count:
                break
    return selected[:purchase_count]


def prize_rank(main_match: int, bonus_match: int) -> str:
    return f"{main_match}/7"


def evaluate_combo(combo: Sequence[int], draw: Draw) -> Tuple[int, int, str]:
    s = set(combo)
    main_match = len(s & set(draw.main))
    bonus_match = len(s & set(draw.bonus)) if draw.bonus else 0
    return main_match, bonus_match, prize_rank(main_match, bonus_match)


def git_commit_push(message: str, paths: Sequence[str], retries: int = 3) -> bool:
    if os.environ.get("DISABLE_GIT_PUSH", "").lower() in {"1", "true", "yes"}:
        print("[GIT] DISABLE_GIT_PUSH ukljucen. Preskacem commit/push.")
        return False
    if not Path(".git").exists():
        print("[GIT] .git nije pronadjen. Preskacem commit/push.")
        return False

    existing_paths = [p for p in paths if Path(p).exists()]
    if not existing_paths:
        print("[GIT] nema fajlova za dodavanje. Preskacem.")
        return False

    try:
        subprocess.run(["git", "config", "user.name", "github-actions[bot]"], check=False)
        subprocess.run(["git", "config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com"], check=False)
        subprocess.run(["git", "add", "-f", *existing_paths], check=True)
        diff = subprocess.run(["git", "diff", "--cached", "--quiet"], check=False)
        if diff.returncode == 0:
            print("[GIT] nema izmena za commit.")
            return False
        subprocess.run(["git", "commit", "-m", message], check=True)
        for attempt in range(1, retries + 1):
            subprocess.run(["git", "pull", "--rebase", "--autostash"], check=False)
            push = subprocess.run(["git", "push"], check=False)
            if push.returncode == 0:
                print("[GIT] push uspeo.")
                return True
            print(f"[GIT] push neuspesan pokusaj={attempt}. kod={push.returncode}")
            time.sleep(2 * attempt)
    except Exception as exc:
        print(f"[GIT] commit/push greska: {exc}", file=sys.stderr)
    return False


def _safe_unlink(path: str) -> None:
    p = Path(path)
    if p.exists():
        p.unlink()
        print(f"[RESET] obrisan {path}")


def infer_existing_purchase_count(result_csv: str) -> Optional[int]:
    p = Path(result_csv)
    if not p.exists() or p.stat().st_size == 0:
        return None
    try:
        counts: Dict[str, int] = {}
        with p.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                key = str(row.get("target_draw_no", ""))
                if key:
                    counts[key] = counts.get(key, 0) + 1
                if len(counts) >= 5:
                    break
        if not counts:
            return None
        values = list(counts.values())
        return max(set(values), key=values.count)
    except Exception as exc:
        print(f"[UPOZ] ne mogu da procitam purchase_count: {exc}")
        return None


def reset_backtest_outputs(result_csv: str, summary_csv: str, resume_state_path: str, reason: str) -> None:
    print(f"[RESET] ponovno gradim backtest izlaz: {reason}")
    for path in [result_csv, summary_csv, resume_state_path]:
        _safe_unlink(path)


def run_backtest(
    draws: Sequence[Draw],
    output_dir: str,
    resume_state_path: str,
    purchase_count: int,
    min_train_draws: int,
    max_targets: Optional[int],
    push_every: int,
    force_rebuild: bool = False,
) -> Dict[str, object]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result_csv = str(out / "loto7_backtest_result.csv")
    summary_csv = str(out / "loto7_backtest_summary.csv")

    state = load_json(resume_state_path)
    existing_purchase_count = infer_existing_purchase_count(result_csv)

    reset_reason = ""
    if force_rebuild:
        reset_reason = "--force-rebuild"
    elif state:
        state_purchase_count = state.get("purchase_count")
        state_min_train = state.get("min_train_draws")
        if state_purchase_count is not None and int(state_purchase_count) != int(purchase_count):
            reset_reason = f"purchase_count changed: {state_purchase_count} -> {purchase_count}"
        elif state_min_train is not None and int(state_min_train) != int(min_train_draws):
            reset_reason = f"min_train_draws changed: {state_min_train} -> {min_train_draws}"
    elif existing_purchase_count is not None and int(existing_purchase_count) != int(purchase_count):
        reset_reason = f"existing result ticket count changed: {existing_purchase_count} -> {purchase_count}"

    if reset_reason:
        reset_backtest_outputs(result_csv, summary_csv, resume_state_path, reset_reason)
        state = {}

    last_completed = int(state.get("last_completed_draw_no", 0) or 0)
    processed_now = 0
    prize_counts: Dict[str, int] = {}
    max_main_match = 0

    targets = [d for i, d in enumerate(draws) if i >= min_train_draws and d.draw_no > last_completed]
    if max_targets is not None:
        targets = targets[:max_targets]

    print(f"[BACKTEST] ciljeva={len(targets)} zavrseno={last_completed} broj_komb={purchase_count}")

    for target in targets:
        target_index = next(i for i, d in enumerate(draws) if d.draw_no == target.draw_no)
        train = list(draws[:target_index])
        combos = generate_candidates(train, purchase_count=purchase_count)

        rows = []
        for idx, combo in enumerate(combos, start=1):
            main_match, bonus_match, rank = evaluate_combo(combo, target)
            prize_counts[rank] = prize_counts.get(rank, 0) + 1
            max_main_match = max(max_main_match, main_match)
            rows.append(
                {
                    "target_draw_no": target.draw_no,
                    "target_date": target.date,
                    "combo_index": idx,
                    "numbers": " ".join(f"{n:02d}" for n in combo),
                    "actual_main": " ".join(f"{n:02d}" for n in target.main),
                    "actual_bonus": " ".join(f"{n:02d}" for n in target.bonus),
                    "main_match": main_match,
                    "bonus_match": bonus_match,
                    "prize_rank": rank,
                }
            )
        append_csv(
            result_csv,
            ["target_draw_no", "target_date", "combo_index", "numbers", "actual_main", "actual_bonus", "main_match", "bonus_match", "prize_rank"],
            rows,
        )

        processed_now += 1
        save_json(
            resume_state_path,
            {
                "last_completed_draw_no": target.draw_no,
                "last_completed_date": target.date,
                "processed_now": processed_now,
                "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "purchase_count": purchase_count,
                "min_train_draws": min_train_draws,
            },
        )

        if push_every > 0 and processed_now % push_every == 0:
            git_commit_push(
                f"Update LOTO7 backtest progress up to draw {target.draw_no}",
                [result_csv, summary_csv, resume_state_path],
            )

    total_rows = 0
    total_targets = set()
    prize_counts = {}
    max_main_match = 0
    if Path(result_csv).exists():
        with Path(result_csv).open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_rows += 1
                total_targets.add(row.get("target_draw_no", ""))
                rank = row.get("prize_rank", "0/7") or "0/7"
                prize_counts[rank] = prize_counts.get(rank, 0) + 1
                try:
                    max_main_match = max(max_main_match, int(row.get("main_match", 0)))
                except ValueError:
                    pass

    summary_rows = []
    for rank in [f"{i}/7" for i in range(8)]:
        count = prize_counts.get(rank, 0)
        summary_rows.append(
            {
                "metric": rank,
                "value": count,
                "rate": f"{(count / total_rows * 100):.6f}%" if total_rows else "0.000000%",
            }
        )
    summary_rows.extend(
        [
            {"metric": "targets", "value": len(total_targets), "rate": ""},
            {"metric": "tickets", "value": total_rows, "rate": ""},
            {"metric": "purchase_count", "value": purchase_count, "rate": ""},
            {"metric": "min_train_draws", "value": min_train_draws, "rate": ""},
            {"metric": "processed_now", "value": processed_now, "rate": ""},
            {"metric": "max_main_match", "value": max_main_match, "rate": ""},
            {"metric": "updated_at", "value": dt.datetime.now(dt.timezone.utc).isoformat(), "rate": ""},
        ]
    )
    write_csv(summary_csv, ["metric", "value", "rate"], summary_rows)

    return {
        "processed_now": processed_now,
        "targets_total": len(total_targets),
        "tickets_total": total_rows,
        "max_main_match": max_main_match,
        "result_csv": result_csv,
        "summary_csv": summary_csv,
        "force_rebuild": force_rebuild,
    }


def predict_latest(draws: Sequence[Draw], output_dir: str, purchase_count: int) -> str:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    combos = generate_candidates(draws, purchase_count=purchase_count)
    latest = draws[-1]
    next_draw_no = latest.draw_no + 1
    path = str(out / "loto7_latest_prediction.csv")
    rows = []
    for idx, combo in enumerate(combos, start=1):
        rows.append(
            {
                "base_latest_draw_no": latest.draw_no,
                "base_latest_date": latest.date,
                "prediction_draw_no": next_draw_no,
                "combo_index": idx,
                "numbers": " ".join(f"{n:02d}" for n in combo),
                "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            }
        )
    write_csv(path, ["base_latest_draw_no", "base_latest_date", "prediction_draw_no", "combo_index", "numbers", "created_at"], rows)
    print(f"[PREDIKCIJA] upisano {path}")
    for row in rows:
        print(f"  {row['combo_index']}: {row['numbers']}")
    return path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="NEW_LOTO7_v2 — Srpski Loto 7/39 pipeline")
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--output-dir", default=".", help="Dir za CSV izlaz (podrazumevano: root projekta)")
    parser.add_argument("--resume-state", default="resume_state.json")
    parser.add_argument("--purchase-count", type=int, default=1)
    parser.add_argument("--min-train-draws", type=int, default=60)
    parser.add_argument("--max-targets", default=None, help="Broj ciljeva validacije (all/None = svi)")
    parser.add_argument("--push-every", type=int, default=100, help="Commit/push na svakih N kola (0 = iskljuceno)")
    parser.add_argument("--push-final", action="store_true")
    parser.add_argument("--force-rebuild", action="store_true", help="Obrisi postojeci backtest i resume; kreni ispocetka")
    parser.add_argument("--skip-backtest", action="store_true")
    args = parser.parse_args(argv)

    if args.purchase_count <= 0:
        raise SystemExit("--purchase-count mora biti pozitivan")

    max_targets: Optional[int]
    if args.max_targets is None or str(args.max_targets).lower() in {"all", "none", ""}:
        max_targets = None
    else:
        max_targets = int(args.max_targets)

    draws = load_draws(args.csv)
    if len(draws) <= args.min_train_draws:
        raise SystemExit(f"Premalo izvlacenja: {len(draws)} <= min_train_draws={args.min_train_draws}")

    result: Dict[str, object] = {}
    if not args.skip_backtest:
        result = run_backtest(
            draws=draws,
            output_dir=args.output_dir,
            resume_state_path=args.resume_state,
            purchase_count=args.purchase_count,
            min_train_draws=args.min_train_draws,
            max_targets=max_targets,
            push_every=args.push_every,
            force_rebuild=args.force_rebuild,
        )
        print(f"[SUMMARY] {json.dumps(result, ensure_ascii=False)}")

    prediction_path = predict_latest(draws, output_dir=args.output_dir, purchase_count=args.purchase_count)

    from export_next_v2 import export_next_v2
    export_next_v2(args.csv, "next_v2.txt")

    if args.push_final:
        paths = [
            args.csv,
            args.resume_state,
            str(Path(args.output_dir) / "loto7_backtest_result.csv"),
            str(Path(args.output_dir) / "loto7_backtest_summary.csv"),
            prediction_path,
        ]
        git_commit_push("Azuriranje LOTO7 pipeline izlaza", paths)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())



"""
[PREDIKCIJA] upisano loto7_latest_prediction.csv
  1: 01 x 14 y 24 z 38
[REZIM] PUNA OPTIMIZACIJA | pool=24 | seed=39
[A] Pipeline (logika)...
[B] Napredni optimizer (Optuna + MC + MCTS)...





2. Brzi next_v2.txt (~2–5 min):

export_next_v2.py --fast 2>&1 | tee next_v2_fast.log


3. Srednji (bolji od --fast, ~30–60 min):

cd ...
export LOTO7_DISABLE_OPTIMIZE=1
export LOTO7_MONTE_CARLO=2000
export LOTO7_MCTS_ITERATIONS=800
export LOTO7_SKIP_RECENT=0
PYTHONUNBUFFERED=1 /python export_next_v2.py 2>&1 | tee next_v2_mid.log

Pipeline backtest ostaje — ne dira ga ovo. Izlaz: next_v2.txt.


[REZIM] PUNA OPTIMIZACIJA | pool=24 | seed=39
[A] Pipeline (logika)...
[B] Napredni optimizer (Optuna + MC + MCTS)...
"""



"""
Skripte
Fajl	Šta radi
loto7_pipeline.py
Glavni: backtest (walk-forward) + Model A + poziva export_next_v2
loto7_logic_predictor.py
Model B predikcija + advanced backtest (spor, opciono)
loto7_advanced_optimizer.py
Model B — Optuna, MC, MCTS (biblioteka, ne pokrećeš direktno)
loto7_enhanced_predictor.py
Model C — obrasci iz backtest detalja
export_next_v2.py
Finalni izlaz → next_v2.txt (3 modela A/B/C)
requirements.txt
Paketi (numpy, sklearn, optuna, torch…)

Izlazi
Fajl	Šta je
next_v2.txt
Predikcije za sledeće kolo (stari 6-komb format dok ne završi novi export)
loto7_latest_prediction.csv
Model A — 1 kombinacija za kolo 4635
loto7_backtest_result.csv
Pipeline backtest — 4574 kola, gotovo
loto7_backtest_summary.csv
Statistika backtesta (prosek pogodaka…)
resume_state.json
Nastavak pipeline backtesta ako prekineš


u praksi
python export_next_v2.py --fast          # brzo
python loto7_pipeline.py                 # dopuna backtest + A

Podaci: data/loto7hh_4634_k48.csv
"""




"""
python export_next_v2.py


next_v2.txt

NEW_LOTO7_v2 — next predikcija (6 kombinacija = 3×2)
CSV: /Users/4c/Desktop/GHQ/data/loto7hh_4634_k48.csv
Poslednje kolo: 4634 | Sledece: 4635
Seed: 39

Model A — Pipeline (logika: freq + pair + structure)
  1: 01 08 14 16 24 34 38
  2: 01 11 14 16 24 31 34

Model B — Advanced Optimizer
  1: 01 03 08 09 30 31 38
  2: 01 09 16 19 24 31 38

Model C — Enhanced Predictor
  1: 01 03 08 09 30 31 38
  2: 01 03 16 19 24 31 38

FINALNE 6 kombinacija (redom A1 A2 B1 B2 C1 C2):
  1: 01 08 14 16 24 34 38
  2: 01 11 14 16 24 31 34
  3: 01 03 08 09 30 31 38
  4: 01 09 16 19 24 31 38
  5: 01 03 08 09 30 31 38
  6: 01 03 16 19 24 31 38
"""








"""
cd /Users/4c/Desktop/GHQ/STATISTIKA/NEW_LOTO7_v2-main
1 — pipeline backtest (ceo CSV, nastavlja ako prekine):

PYTHONUNBUFFERED=1 /Users/4c/tesla_env/bin/python loto7_pipeline.py --max-targets all --push-every 0 2>&1 | tee pipeline.log
Ispočetka (briše stari backtest):

PYTHONUNBUFFERED=1 /Users/4c/tesla_env/bin/python loto7_pipeline.py --force-rebuild --max-targets all --push-every 0 2>&1 | tee pipeline.log
2 — advanced backtest + predikcija + next_v2.txt:

PYTHONUNBUFFERED=1 /Users/4c/tesla_env/bin/python loto7_logic_predictor.py --backtest --max-backtest-draws 0 2>&1 | tee logic_backtest.log
Samo finalni izvoz (ako backtest već gotov):

PYTHONUNBUFFERED=1 /Users/4c/tesla_env/bin/python export_next_v2.py 2>&1 | tee next_v2.log
Preko noći u pozadini (redom 1 pa 2):

nohup bash -c '
cd /Users/4c/Desktop/GHQ/STATISTIKA/NEW_LOTO7_v2-main
PYTHONUNBUFFERED=1 /Users/4c/tesla_env/bin/python loto7_pipeline.py --max-targets all --push-every 0
PYTHONUNBUFFERED=1 /Users/4c/tesla_env/bin/python loto7_logic_predictor.py --backtest --max-backtest-draws 0
' > overnight.log 2>&1 &
Izlaz: next_v2.txt (+ CSV backtest fajlovi). Bez --fast.
"""

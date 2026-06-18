#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
loto7_enhanced_predictor.py

Poboljsani prediktor za v2.

    - score brojeva sa vremenskim opadanjem
    - par/trojka korelacije sa opadanjem
    - ponovno koriscenje visokih pogodaka iz backtest detalja
    - bez future leak-a (samo istorija pre ciljnog kola)
    - ne menja loto7_logic_predictor.py

Score je rang kandidata, ne verovatnoca.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from loto7_logic_predictor import (
    DEFAULT_CSV,
    Draw,
    TicketScore,
    _csv_kolo,
    _csv_pred,
    _csv_score,
    _csv_strategija,
    count_main_matches,
    format_ticket,
    load_draws,
    prediction_csv_header,
    score_normalized_values,
)

NUM_MIN = 1
NUM_MAX = 39
PICK_SIZE = 7

DEFAULT_ENHANCED_OUTPUT_CSV = "loto7_enhanced_predictions.csv"
DEFAULT_ENHANCED_LATEST_TXT = "latest_loto7_enhanced_prediction.txt"
DEFAULT_ENHANCED_BACKTEST_SUMMARY_CSV = "loto7_enhanced_backtest_summary.csv"
DEFAULT_ENHANCED_BACKTEST_DETAIL_CSV = "loto7_enhanced_backtest_detail.csv"
DEFAULT_HIT_PATTERN_CSV = "loto7_backtest_detail.csv"

LOW_RANGE = range(1, 14)
MID_RANGE = range(14, 27)
HIGH_RANGE = range(27, 40)


def parse_ticket(value: object) -> Tuple[int, ...]:
    import re

    nums = tuple(int(x) for x in re.findall(r"\d+", str(value or "")))
    nums = tuple(sorted(n for n in nums if NUM_MIN <= n <= NUM_MAX))
    if len(nums) == PICK_SIZE and len(set(nums)) == PICK_SIZE:
        return nums
    return tuple()


def band_counts(ticket: Sequence[int]) -> Tuple[int, int, int]:
    low = sum(1 for n in ticket if n in LOW_RANGE)
    mid = sum(1 for n in ticket if n in MID_RANGE)
    high = sum(1 for n in ticket if n in HIGH_RANGE)
    return low, mid, high


def max_consecutive_run(ticket: Sequence[int]) -> int:
    nums = sorted(ticket)
    if not nums:
        return 0
    best = cur = 1
    for a, b in zip(nums, nums[1:]):
        if b == a + 1:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best


def decay_weight(age: int, half_life: float) -> float:
    if half_life <= 0:
        return 1.0
    return 0.5 ** (age / half_life)


def weighted_number_counts(draws: Sequence[Draw], half_life: float, window: int = 0, bonus: bool = False) -> Counter:
    target_draws = list(draws[-window:]) if window and window > 0 else list(draws)
    latest_index = len(target_draws) - 1
    counter: Counter = Counter()

    for idx, draw in enumerate(target_draws):
        age = latest_index - idx
        weight = decay_weight(age, half_life)
        nums = draw.bonus if bonus else draw.main
        for n in nums:
            counter[n] += weight

    return counter


def weighted_combination_counts(draws: Sequence[Draw], k: int, half_life: float, window: int = 0) -> Counter:
    target_draws = list(draws[-window:]) if window and window > 0 else list(draws)
    latest_index = len(target_draws) - 1
    counter: Counter = Counter()

    for idx, draw in enumerate(target_draws):
        age = latest_index - idx
        weight = decay_weight(age, half_life)
        for comb in itertools.combinations(sorted(draw.main), k):
            counter[comb] += weight

    return counter


def counter_norm(counter: Counter, key: object) -> float:
    if not counter:
        return 0.0
    max_value = max(counter.values()) or 1.0
    return float(counter.get(key, 0.0)) / float(max_value)


def last_seen_gaps(draws: Sequence[Draw]) -> Dict[int, int]:
    last: Dict[int, Optional[int]] = {n: None for n in range(NUM_MIN, NUM_MAX + 1)}
    for idx, draw in enumerate(draws):
        for n in draw.main:
            last[n] = idx

    latest = len(draws) - 1
    return {
        n: (len(draws) + 1 if idx is None else latest - int(idx))
        for n, idx in last.items()
    }


def build_number_scores(draws: Sequence[Draw]) -> Dict[int, float]:
    c_short = weighted_number_counts(draws, half_life=10.0, window=40)
    c_mid = weighted_number_counts(draws, half_life=28.0, window=120)
    c_long = weighted_number_counts(draws, half_life=75.0, window=0)
    bonus_mid = weighted_number_counts(draws, half_life=24.0, window=120, bonus=True)
    gaps = last_seen_gaps(draws)

    scores: Dict[int, float] = {}
    for n in range(NUM_MIN, NUM_MAX + 1):
        hot = (
            4.4 * counter_norm(c_short, n)
            + 2.4 * counter_norm(c_mid, n)
            + 1.0 * counter_norm(c_long, n)
        )
        bonus = 0.18 * counter_norm(bonus_mid, n)

        gap = gaps[n]
        if gap == 0:
            gap_score = -0.22
        elif 2 <= gap <= 18:
            gap_score = 0.12 + min(gap, 18) / 90.0
        elif 19 <= gap <= 30:
            gap_score = 0.08
        else:
            gap_score = -0.03

        scores[n] = hot + bonus + gap_score

    return scores


def load_hit_patterns(
    path: str = DEFAULT_HIT_PATTERN_CSV,
    min_matches: int = 4,
    max_patterns: int = 300,
    before_date: Optional[str] = None,
) -> List[Tuple[int, ...]]:
    csv_path = Path(path)
    if not csv_path.exists():
        return []

    patterns: List[Tuple[int, ...]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            date = row.get("kolo") or row.get("抽せん日", "")
            if before_date and date >= before_date:
                continue

            for i in range(1, 51):
                pred_key = f"predikcija{i}"
                match_key = f"predikcija{i}_pogodaka"
                legacy_pk = f"予測{i}"
                if pred_key not in row and legacy_pk not in row:
                    break
                try:
                    matches = int(str(row.get(match_key) or row.get(f"予測{i}_本数字一致", "0") or "0"))
                except ValueError:
                    matches = 0
                if matches >= min_matches:
                    ticket = parse_ticket(row.get(pred_key) or row.get(legacy_pk, ""))
                    if ticket:
                        patterns.append(ticket)

    # Novija istorija ima prioritet; uklanjanje duplikata
    deduped: List[Tuple[int, ...]] = []
    seen = set()
    for ticket in reversed(patterns):
        if ticket not in seen:
            deduped.append(ticket)
            seen.add(ticket)
        if len(deduped) >= max_patterns:
            break
    return list(reversed(deduped))


def pattern_similarity(ticket: Sequence[int], patterns: Sequence[Sequence[int]]) -> float:
    if not patterns:
        return 0.0
    t = set(ticket)
    best = 0.0
    total = 0.0
    recent_weight_total = 0.0

    for idx, pattern in enumerate(patterns):
        age = len(patterns) - 1 - idx
        weight = decay_weight(age, half_life=80.0)
        overlap = len(t & set(pattern))
        # Struktura slicna proslim kandidatima sa 4+ pogotka; puno kopiranje blago kaznjeno
        if overlap >= 6:
            sim = 0.35
        elif overlap == 5:
            sim = 0.75
        elif overlap == 4:
            sim = 0.45
        else:
            sim = 0.0
        best = max(best, sim)
        total += weight * sim
        recent_weight_total += weight

    avg = total / recent_weight_total if recent_weight_total else 0.0
    return 0.65 * best + 0.35 * avg


def structure_penalty(ticket: Sequence[int], draws: Sequence[Draw]) -> float:
    ticket = tuple(sorted(ticket))
    last_main = draws[-1].main
    prev_main = draws[-2].main if len(draws) >= 2 else tuple()

    low, mid, high = band_counts(ticket)
    odd = sum(1 for n in ticket if n % 2 == 1)
    total = sum(ticket)
    repeat_last = len(set(ticket) & set(last_main))
    repeat_prev = len(set(ticket) & set(prev_main))
    run = max_consecutive_run(ticket)
    last_digits = [n % 10 for n in ticket]
    last_digit_dup = max(Counter(last_digits).values())

    penalty = 0.0

    if odd not in (3, 4):
        penalty += 0.25 if odd in (2, 5) else 0.75

    if not (2 <= low <= 3):
        penalty += 0.35 * abs(low - 2.5)
    if not (2 <= mid <= 3):
        penalty += 0.35 * abs(mid - 2.5)
    if not (1 <= high <= 3):
        penalty += 0.25 * abs(high - 2.0)

    if total < 115:
        penalty += (115 - total) / 28.0
    elif total > 175:
        penalty += (total - 175) / 28.0

    if run >= 4:
        penalty += 1.10
    elif run == 3:
        penalty += 0.28

    if repeat_last == 0:
        penalty += 0.55
    elif repeat_last == 1:
        penalty += 0.08
    elif 2 <= repeat_last <= 4:
        penalty += 0.0
    elif repeat_last == 5:
        penalty += 0.38
    else:
        penalty += 0.85

    if repeat_prev >= 5:
        penalty += 0.35

    if last_digit_dup >= 4:
        penalty += 0.55
    elif last_digit_dup == 3:
        penalty += 0.18

    return penalty


def enhanced_ticket_score(
    ticket: Sequence[int],
    draws: Sequence[Draw],
    number_scores: Dict[int, float],
    pair_short: Counter,
    pair_mid: Counter,
    pair_long: Counter,
    triple_mid: Counter,
    triple_long: Counter,
    hit_patterns: Sequence[Sequence[int]],
    strategy: str,
) -> TicketScore:
    ticket = tuple(sorted(ticket))
    single = sum(number_scores[n] for n in ticket)

    pair_score = 0.0
    for pair in itertools.combinations(ticket, 2):
        pair_score += 0.42 * counter_norm(pair_short, pair)
        pair_score += 0.25 * counter_norm(pair_mid, pair)
        pair_score += 0.10 * counter_norm(pair_long, pair)

    triple_score = 0.0
    for tri in itertools.combinations(ticket, 3):
        triple_score += 0.52 * counter_norm(triple_mid, tri)
        triple_score += 0.18 * counter_norm(triple_long, tri)

    pattern_score = 1.05 * pattern_similarity(ticket, hit_patterns)
    penalty = structure_penalty(ticket, draws)

    low, mid, high = band_counts(ticket)
    strategy_bonus = 0.0
    if strategy == "BALANCED" and low >= 2 and mid >= 2 and high >= 2:
        strategy_bonus += 0.42
    elif strategy == "HIGH_BIAS" and high >= 3:
        strategy_bonus += 0.34
    elif strategy == "LOW_BIAS" and low >= 3:
        strategy_bonus += 0.34
    elif strategy == "PATTERN" and pattern_score > 0:
        strategy_bonus += 0.30
    elif strategy == "COLD":
        gaps = last_seen_gaps(draws)
        strategy_bonus += sum(min(gaps[n], 35) for n in ticket) / 260.0

    total_score = single + pair_score + triple_score + pattern_score + strategy_bonus - penalty

    return TicketScore(
        ticket=ticket,
        score=total_score,
        strategy=strategy,
        detail={
            "single": single,
            "pair": pair_score,
            "triple": triple_score,
            "pattern": pattern_score,
            "strategy_bonus": strategy_bonus,
            "penalty": penalty,
            "sum": float(sum(ticket)),
            "odd": float(sum(1 for n in ticket if n % 2 == 1)),
            "low": float(low),
            "mid": float(mid),
            "high": float(high),
            "repeat_last": float(len(set(ticket) & set(draws[-1].main))),
        },
    )


def make_candidate_pool(draws: Sequence[Draw], pool_size: int = 20) -> List[int]:
    number_scores = build_number_scores(draws)
    pair_mid = weighted_combination_counts(draws, 2, half_life=28.0, window=140)
    gaps = last_seen_gaps(draws)

    base_rank = sorted(range(NUM_MIN, NUM_MAX + 1), key=lambda n: number_scores[n], reverse=True)
    pool = list(base_rank[: max(12, pool_size - 4)])

    # Brojevi iz jakih parova
    for pair, _ in pair_mid.most_common(40):
        for n in pair:
            if n not in pool:
                pool.append(n)
            if len(pool) >= pool_size:
                break
        if len(pool) >= pool_size:
            break

    # Malo dugo odsutnih brojeva
    for n, _ in sorted(gaps.items(), key=lambda kv: kv[1], reverse=True):
        if n not in pool:
            pool.append(n)
        if len(pool) >= pool_size + 2:
            break

    return sorted(pool[: max(pool_size, PICK_SIZE)])


def select_diverse_tickets(
    ranked: Sequence[TicketScore],
    num_tickets: int,
    max_overlap: int = 4,
    max_number_usage: int = 3,
) -> List[TicketScore]:
    selected: List[TicketScore] = []
    usage: Counter = Counter()

    for item in ranked:
        ticket_set = set(item.ticket)
        if any(len(ticket_set & set(chosen.ticket)) > max_overlap for chosen in selected):
            continue
        if any(usage[n] >= max_number_usage for n in item.ticket):
            continue
        selected.append(item)
        usage.update(item.ticket)
        if len(selected) >= num_tickets:
            return selected

    # Ako fali, olabavi ogranicenja
    for item in ranked:
        if item in selected:
            continue
        if any(len(set(item.ticket) & set(chosen.ticket)) > 5 for chosen in selected):
            continue
        selected.append(item)
        if len(selected) >= num_tickets:
            return selected

    return selected[:num_tickets]


def enhanced_predict(
    draws: Sequence[Draw],
    num_tickets: int = 10,
    pool_size: int = 20,
    hit_pattern_csv: str = DEFAULT_HIT_PATTERN_CSV,
    before_date: Optional[str] = None,
) -> List[TicketScore]:
    if len(draws) < 2:
        raise ValueError("Za predikciju treba bar 2 izvlacenja.")

    pool = make_candidate_pool(draws, pool_size=pool_size)
    number_scores = build_number_scores(draws)
    pair_short = weighted_combination_counts(draws, 2, half_life=10.0, window=60)
    pair_mid = weighted_combination_counts(draws, 2, half_life=28.0, window=160)
    pair_long = weighted_combination_counts(draws, 2, half_life=80.0, window=0)
    triple_mid = weighted_combination_counts(draws, 3, half_life=30.0, window=180)
    triple_long = weighted_combination_counts(draws, 3, half_life=95.0, window=0)
    hit_patterns = load_hit_patterns(hit_pattern_csv, before_date=before_date)

    ranked: List[TicketScore] = []
    for ticket in itertools.combinations(pool, PICK_SIZE):
        low, mid, high = band_counts(ticket)
        strategies = ["CORE"]
        if low >= 2 and mid >= 2 and high >= 2:
            strategies.append("BALANCED")
        if high >= 3:
            strategies.append("HIGH_BIAS")
        if low >= 3:
            strategies.append("LOW_BIAS")
        if hit_patterns:
            strategies.append("PATTERN")

        best: Optional[TicketScore] = None
        for strategy in strategies:
            scored = enhanced_ticket_score(
                ticket,
                draws,
                number_scores,
                pair_short,
                pair_mid,
                pair_long,
                triple_mid,
                triple_long,
                hit_patterns,
                strategy=strategy,
            )
            if best is None or scored.score > best.score:
                best = scored
        if best is not None:
            ranked.append(best)

    ranked.sort(key=lambda x: x.score, reverse=True)
    return select_diverse_tickets(ranked, num_tickets=num_tickets)


def save_predictions_csv(path: str, target_date: str, tickets: Sequence[TicketScore], save_count: int) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = prediction_csv_header(save_count)
    scores = score_normalized_values(tickets[:save_count])
    new_row = {key: "" for key in header}
    new_row["kolo"] = target_date

    for i, ticket in enumerate(tickets[:save_count], start=1):
        new_row[f"predikcija{i}"] = format_ticket(ticket.ticket, zero_pad=False)
        new_row[f"score{i}"] = f"{scores[i - 1]:.3f}".rstrip("0").rstrip(".")
        new_row[f"strategija{i}"] = ticket.strategy

    rows: List[Dict[str, str]] = []
    if out.exists():
        with out.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                mapped = {key: "" for key in header}
                mapped["kolo"] = _csv_kolo(row)
                for i in range(1, save_count + 1):
                    mapped[f"predikcija{i}"] = _csv_pred(row, i)
                    mapped[f"score{i}"] = _csv_score(row, i)
                    mapped[f"strategija{i}"] = _csv_strategija(row, i)
                rows.append(mapped)

    rows = [row for row in rows if _csv_kolo(row) != target_date]
    rows.append(new_row)
    rows.sort(key=lambda r: _csv_kolo(r))

    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def save_latest_txt(path: str, target_date: str, tickets: Sequence[TicketScore]) -> None:
    scores = score_normalized_values(tickets)
    lines = [
        "Loto7 — poboljsana predikcija",
        "==============================",
        f"Ciljno kolo: {target_date}",
        "",
        "Predikcije",
        "--------------------------",
    ]

    for i, ticket in enumerate(tickets, start=1):
        detail = ticket.detail
        lines.append(
            f"{i:02d}. {format_ticket(ticket.ticket, zero_pad=False)}"
            f" / score: {scores[i - 1]:.3f}".rstrip("0").rstrip(".")
            + f" / strategija: {ticket.strategy}"
            + f" / par: {detail.get('pair', 0):.3f}"
            + f" / trojka: {detail.get('triple', 0):.3f}"
            + f" / obrazac: {detail.get('pattern', 0):.3f}"
        )

    lines.extend([
        "",
        "Napomena",
        "--------------------------",
        "Score je rang kandidata, ne verovatnoca dobitka.",
        "Loto je nezavisno izvlacenje; nema garancije pogotka.",
    ])

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def enhanced_backtest(
    draws: Sequence[Draw],
    min_train: int,
    num_tickets: int,
    pool_size: int,
    hit_pattern_csv: str,
    max_backtest_draws: int,
    summary_csv: str,
    detail_csv: str,
) -> Dict[str, object]:
    if len(draws) <= min_train:
        raise ValueError("Za backtest treba vise podataka od min_train.")

    start = max(min_train, len(draws) - max_backtest_draws) if max_backtest_draws > 0 else min_train
    detail_rows: List[Dict[str, object]] = []
    top1_hits: List[int] = []
    best_hits: List[int] = []

    for i in range(start, len(draws)):
        train = draws[:i]
        actual = draws[i]
        tickets = enhanced_predict(
            train,
            num_tickets=num_tickets,
            pool_size=pool_size,
            hit_pattern_csv=hit_pattern_csv,
            before_date=actual.date,
        )
        hits = [count_main_matches(t.ticket, actual.main) for t in tickets]
        top1_hits.append(hits[0] if hits else 0)
        best_hits.append(max(hits) if hits else 0)

        row: Dict[str, object] = {
            "kolo": actual.date,
            "broj_kola": actual.draw_no or "",
            "izvuceno": format_ticket(actual.main),
            "max_pogodaka": max(hits) if hits else 0,
        }
        for idx, (ticket, h) in enumerate(zip(tickets, hits), start=1):
            row[f"predikcija{idx}"] = format_ticket(ticket.ticket)
            row[f"predikcija{idx}_strategija"] = ticket.strategy
            row[f"predikcija{idx}_pogodaka"] = h
        detail_rows.append(row)

    def rate(values: Sequence[int], threshold: int) -> float:
        return sum(1 for v in values if v >= threshold) / len(values) if values else 0.0

    summary = {
        "broj_validacija": len(top1_hits),
        "pocetna_obuka": min_train,
        "pocetak_validacije": start + 1,
        "komb_po_kolu": num_tickets,
        "velicina_pool": pool_size,
        "prosek_pog_prva": round(sum(top1_hits) / len(top1_hits), 6) if top1_hits else 0,
        "prosek_pog_max": round(sum(best_hits) / len(best_hits), 6) if best_hits else 0,
        "stopa_prva_3plus": round(rate(top1_hits, 3), 6),
        "stopa_prva_4plus": round(rate(top1_hits, 4), 6),
        "stopa_max_3plus": round(rate(best_hits, 3), 6),
        "stopa_max_4plus": round(rate(best_hits, 4), 6),
        "distribucija_prva": dict(sorted(Counter(top1_hits).items())),
        "distribucija_max": dict(sorted(Counter(best_hits).items())),
    }

    summary_path = Path(summary_csv)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    detail_path = Path(detail_csv)
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    if detail_rows:
        fieldnames = list(detail_rows[0].keys())
        with detail_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(detail_rows)

    return summary


def print_predictions(tickets: Sequence[TicketScore]) -> None:
    scores = score_normalized_values(tickets)
    print("=== Loto7 — poboljsana predikcija ===")
    for i, ticket in enumerate(tickets, start=1):
        d = ticket.detail
        print(
            f"{i:02d}. {format_ticket(ticket.ticket)}"
            f" | score={scores[i - 1]:.3f}"
            f" | strategija={ticket.strategy}"
            f" | par={d.get('pair', 0):.3f}"
            f" | trojka={d.get('triple', 0):.3f}"
            f" | obrazac={d.get('pattern', 0):.3f}"
            f" | suma={int(d.get('sum', 0))}"
            f" | parno={int(d.get('odd', 0))}"
            f" | nizak/srednji/visok={int(d.get('low', 0))}/{int(d.get('mid', 0))}/{int(d.get('high', 0))}"
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Loto7 poboljsana predikcija i backtest")
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--tickets", type=int, default=10)
    parser.add_argument("--pool-size", type=int, default=20)
    parser.add_argument("--output-csv", default=DEFAULT_ENHANCED_OUTPUT_CSV)
    parser.add_argument("--latest-txt", default=DEFAULT_ENHANCED_LATEST_TXT)
    parser.add_argument("--save-count", type=int, default=10)
    parser.add_argument("--hit-pattern-csv", default=DEFAULT_HIT_PATTERN_CSV)
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument("--min-train", type=int, default=100)
    parser.add_argument("--max-backtest-draws", type=int, default=120)
    parser.add_argument("--backtest-summary-csv", default=DEFAULT_ENHANCED_BACKTEST_SUMMARY_CSV)
    parser.add_argument("--backtest-detail-csv", default=DEFAULT_ENHANCED_BACKTEST_DETAIL_CSV)
    args = parser.parse_args(argv)

    draws = load_draws(args.csv)
    if not draws:
        raise SystemExit("Nije moguce ucitati CSV izvlacenja.")

    target_date = str((draws[-1].draw_no or len(draws)) + 1)
    tickets = enhanced_predict(
        draws,
        num_tickets=args.tickets,
        pool_size=args.pool_size,
        hit_pattern_csv=args.hit_pattern_csv,
    )
    print_predictions(tickets)

    save_predictions_csv(args.output_csv, target_date, tickets, save_count=args.save_count)
    save_latest_txt(args.latest_txt, target_date, tickets[: args.save_count])
    print(f"Sacuvano: {args.output_csv}")
    print(f"Sacuvano: {args.latest_txt}")

    if args.backtest:
        summary = enhanced_backtest(
            draws,
            min_train=args.min_train,
            num_tickets=args.tickets,
            pool_size=args.pool_size,
            hit_pattern_csv=args.hit_pattern_csv,
            max_backtest_draws=args.max_backtest_draws,
            summary_csv=args.backtest_summary_csv,
            detail_csv=args.backtest_detail_csv,
        )
        print("=== Poboljsani backtest — rezultat ===")
        for key, value in summary.items():
            print(f"{key}: {value}")

    print("Napomena: score je rang kandidata, ne verovatnoca. Nema garancije pogotka.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

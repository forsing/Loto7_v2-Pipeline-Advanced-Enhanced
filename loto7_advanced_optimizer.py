#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
loto7_advanced_optimizer.py

Napredni Loto7 optimizer za v2.

Funkcije:
- bodovanje za 3+ / 4+ / 5+ / 6+ pogodaka
- walk-forward validacija bez future leak-a
- Optuna optimizacija sa random-search rezervom
- Monte Carlo + lagani MCTS
- MemoryBank iz proslih visokih pogodaka
- CatBoost meta-regresor sa deterministickim rezervnim putem
- nastavljivi backtest i snimanje napretka

Score je rang kandidata, ne verovatnoca.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import os
import random
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from loto7_logic_predictor import (
    Draw,
    TicketScore,
    count_main_matches,
    format_ticket,
    score_normalized_values,
)
from loto7_enhanced_predictor import (
    DEFAULT_HIT_PATTERN_CSV,
    band_counts,
    build_number_scores,
    counter_norm,
    make_candidate_pool,
    max_consecutive_run,
    save_latest_txt,
    structure_penalty,
    weighted_combination_counts,
)

NUM_MIN = 1
NUM_MAX = 39
PICK_SIZE = 7
SEED = 39
MAX_MEMORYBANK_SIZE = 500

WEIGHTS_CACHE = "loto7_advanced_weights.json"
MEMORYBANK_CSV = "loto7_memorybank.csv"
MEMORYBANK_5PLUS_CSV = "loto7_memorybank_5plus.csv"
MEMORYBANK_4PLUS_CSV = "loto7_memorybank_4plus.csv"
MEMORYBANK_6HIT_CSV = "loto7_memorybank_6hit.csv"
META_REGRESSOR_JSON = "loto7_meta_regressor.json"
CLUSTER_CSV = "loto7_hit_structure_clusters.csv"
RESUME_JSON = "loto7_backtest_resume.json"
NEXTGEN_META6_JSON = "loto7_meta6_classifier.json"
NEXTGEN_SHAP_JSON = "loto7_shap_feature_selection.json"


@dataclass(frozen=True)
class AdvancedWeights:
    single: float = 1.00
    pair: float = 1.15
    triple: float = 1.25
    memory: float = 1.45
    grade6: float = 1.75
    structure: float = 1.00
    diversity: float = 0.25
    memory5: float = 1.60
    meta: float = 0.45
    cluster: float = 0.25
    cycle: float = 0.55
    diffusion: float = 0.35
    ppo: float = 0.35
    meta6: float = 0.85
    shap: float = 0.25
    recent: float = 0.85
    pair_stability: float = 0.65
    pair_recency: float = 0.70
    constraint: float = 0.95
    ensemble: float = 0.45

    def to_dict(self) -> Dict[str, float]:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "AdvancedWeights":
        base = cls().to_dict()
        for k in base:
            try:
                base[k] = float(d.get(k, base[k]))
            except Exception:
                pass
        return cls(**base)


def _parse_ticket(v: object) -> Tuple[int, ...]:
    nums = tuple(sorted(int(x) for x in re.findall(r"\d+", str(v or ""))))
    nums = tuple(n for n in nums if NUM_MIN <= n <= NUM_MAX)
    return nums if len(nums) == PICK_SIZE and len(set(nums)) == PICK_SIZE else tuple()


def _grade_label(g: Optional[int]) -> str:
    return "promasaj" if g is None else f"{g} pogodaka"


def _avg(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _load_weights(path: str = WEIGHTS_CACHE) -> Optional[AdvancedWeights]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return AdvancedWeights.from_dict(data.get("weights", data))
    except Exception:
        return None


def _save_weights(w: AdvancedWeights, path: str = WEIGHTS_CACHE) -> None:
    Path(path).write_text(
        json.dumps({"weights": w.to_dict()}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def ticket_features(ticket: Sequence[int], draws: Sequence[Draw]) -> Dict[str, float]:
    t = tuple(sorted(ticket))
    low, mid, high = band_counts(t)
    odd = sum(n % 2 for n in t)
    total = sum(t)
    run = max_consecutive_run(t)
    last = set(draws[-1].main) if draws else set()
    prev = set(draws[-2].main) if len(draws) >= 2 else set()
    gaps = [b - a for a, b in zip(t, t[1:])]
    last_digits = Counter(n % 10 for n in t)
    decades = Counter(n // 10 for n in t)
    sum_center = 135.0
    sum_band_score = max(0.0, 1.0 - abs(float(total) - sum_center) / 55.0)
    odd_balance_score = 1.0 if odd in (3, 4) else 0.45 if odd in (2, 5) else 0.0
    low_balance_score = 1.0 if low in (3, 4) else 0.50 if low in (2, 5) else 0.0
    return {
        "sum": float(total),
        "odd": float(odd),
        "low": float(low),
        "mid": float(mid),
        "high": float(high),
        "run": float(run),
        "repeat_last": float(len(set(t) & last)),
        "repeat_prev": float(len(set(t) & prev)),
        "last_digit_max": float(max(last_digits.values()) if last_digits else 0),
        "decade_count": float(len(decades)),
        "gap_avg": float(sum(gaps) / len(gaps) if gaps else 0),
        "gap_min": float(min(gaps) if gaps else 0),
        "gap_max": float(max(gaps) if gaps else 0),
        "sum_band_score": float(sum_band_score),
        "odd_balance_score": float(odd_balance_score),
        "low_balance_score": float(low_balance_score),
        "constraint_score": float(0.45 * sum_band_score + 0.35 * odd_balance_score + 0.20 * low_balance_score),
    }


class MemoryBank:
    """Opsta memorija za 4+ pogodaka."""

    def __init__(self) -> None:
        self.items: List[Tuple[Tuple[int, ...], float]] = []
        self.pairs: Counter = Counter()
        self.triples: Counter = Counter()
        self.sums: Counter = Counter()

    def add(self, ticket: Sequence[int], strength: float = 1.0) -> None:
        t = tuple(sorted(ticket))
        if len(t) != PICK_SIZE or len(set(t)) != PICK_SIZE:
            return
        self.items.append((t, float(strength)))
        for p in itertools.combinations(t, 2):
            self.pairs[p] += strength
        for tri in itertools.combinations(t, 3):
            self.triples[tri] += strength
        self.sums[sum(t) // 10 * 10] += strength

    def load_detail(self, path: str, before_date: Optional[str] = None, min_matches: int = 4) -> None:
        p = Path(path)
        if not p.exists():
            return
        try:
            rows = list(csv.DictReader(p.open("r", encoding="utf-8-sig", newline="")))
        except Exception:
            return
        for row in rows:
            date = row.get("kolo") or row.get("抽せん日", "")
            if before_date and date >= before_date:
                continue
            for i in range(1, 101):
                pk = f"predikcija{i}"
                mk = f"predikcija{i}_pogodaka"
                legacy_pk = f"予測{i}"
                if pk not in row and legacy_pk not in row:
                    break
                try:
                    m = int(row.get(mk) or row.get(f"予測{i}_本数字一致", 0) or 0)
                except Exception:
                    m = 0
                if m >= min_matches:
                    t = _parse_ticket(row.get(pk) or row.get(legacy_pk, ""))
                    if t:
                        self.add(t, 1.0 + max(0, m - 4) * 0.8)

    def load_memorybank(self, path: str = MEMORYBANK_CSV, before_date: Optional[str] = None) -> None:
        p = Path(path)
        if not p.exists():
            return
        try:
            for row in csv.DictReader(p.open("r", encoding="utf-8-sig", newline="")):
                if before_date and (row.get("kolo") or row.get("抽せん日", "")) >= before_date:
                    continue
                t = _parse_ticket(row.get("kombinacija") or row.get("組合せ", ""))
                if t:
                    self.add(t, float(row.get("jacina") or row.get("強度", 1) or 1))
        except Exception:
            return

    def save(self, path: str = MEMORYBANK_CSV) -> None:
        with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["kombinacija", "jacina"])
            w.writeheader()
            for t, s in self.items[-1000:]:
                w.writerow({"kombinacija": format_ticket(t), "jacina": round(s, 6)})

    def score(self, ticket: Sequence[int]) -> float:
        if not self.items:
            return 0.0
        s = set(ticket)
        vals = []
        for pat, strength in self.items[-300:]:
            o = len(s & set(pat))
            vals.append(({6: 0.3, 5: 1.0, 4: 0.55, 3: 0.15}.get(o, 0.0)) * strength)
        pair = sum(counter_norm(self.pairs, p) for p in itertools.combinations(sorted(ticket), 2)) / 21.0
        tri = sum(counter_norm(self.triples, t) for t in itertools.combinations(sorted(ticket), 3)) / 35.0
        return 0.60 * max(vals) + 0.20 * _avg(vals) + 0.12 * pair + 0.08 * tri


class MemoryBank5Plus:
    """Posebna memorija samo za 5+ pogodaka."""

    def __init__(self) -> None:
        self.items: List[Tuple[Tuple[int, ...], int, float, str]] = []
        self.pairs: Counter = Counter()
        self.triples: Counter = Counter()
        self.sums: Counter = Counter()

    @staticmethod
    def strength_for_matches(matches: int) -> float:
        if matches >= 7:
            return 8.0
        if matches == 6:
            return 3.5
        if matches == 5:
            return 1.0
        return 0.0

    def add(self, ticket: Sequence[int], matches: int, date: str = "") -> None:
        t = tuple(sorted(ticket))
        if len(t) != PICK_SIZE or len(set(t)) != PICK_SIZE or matches < 5:
            return
        strength = self.strength_for_matches(matches)
        self.items.append((t, int(matches), strength, date))
        for p in itertools.combinations(t, 2):
            self.pairs[p] += strength
        for tri in itertools.combinations(t, 3):
            self.triples[tri] += strength
        self.sums[sum(t) // 10 * 10] += strength

    def load_detail(self, path: str, before_date: Optional[str] = None) -> None:
        p = Path(path)
        if not p.exists():
            return
        try:
            rows = list(csv.DictReader(p.open("r", encoding="utf-8-sig", newline="")))
        except Exception:
            return
        for row in rows:
            date = row.get("kolo") or row.get("抽せん日", "")
            if before_date and date >= before_date:
                continue
            for i in range(1, 101):
                pk = f"predikcija{i}"
                mk = f"predikcija{i}_pogodaka"
                legacy_pk = f"予測{i}"
                if pk not in row and legacy_pk not in row:
                    break
                try:
                    m = int(row.get(mk) or row.get(f"予測{i}_本数字一致", 0) or 0)
                except Exception:
                    m = 0
                if m >= 5:
                    t = _parse_ticket(row.get(pk) or row.get(legacy_pk, ""))
                    if t:
                        self.add(t, m, date)

    def load_memorybank(self, path: str = MEMORYBANK_5PLUS_CSV, before_date: Optional[str] = None) -> None:
        p = Path(path)
        if not p.exists():
            return
        try:
            for row in csv.DictReader(p.open("r", encoding="utf-8-sig", newline="")):
                date = row.get("kolo") or row.get("抽せん日", "")
                if before_date and date >= before_date:
                    continue
                t = _parse_ticket(row.get("kombinacija") or row.get("組合せ", ""))
                try:
                    m = int(row.get("pogodaka") or row.get("一致数", 5) or 5)
                except Exception:
                    m = 5
                if t:
                    self.add(t, m, date)
        except Exception:
            return

    def save(self, path: str = MEMORYBANK_5PLUS_CSV) -> None:
        with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["kolo", "kombinacija", "pogodaka", "jacina"])
            w.writeheader()
            for t, m, s, date in self.items[-1000:]:
                w.writerow({"kolo": date, "kombinacija": format_ticket(t), "pogodaka": m, "jacina": round(s, 6)})

    def score(self, ticket: Sequence[int]) -> float:
        if not self.items:
            return 0.0
        s = set(ticket)
        vals = []
        for pat, _m, strength, _date in self.items[-300:]:
            o = len(s & set(pat))
            vals.append(({7: 0.10, 6: 0.35, 5: 1.15, 4: 0.55, 3: 0.15}.get(o, 0.0)) * strength)
        pair_total = max(sum(self.pairs.values()), 1.0)
        tri_total = max(sum(self.triples.values()), 1.0)
        pair = sum(self.pairs.get(p, 0.0) for p in itertools.combinations(sorted(ticket), 2)) / pair_total
        tri = sum(self.triples.get(t, 0.0) for t in itertools.combinations(sorted(ticket), 3)) / tri_total
        return 0.70 * max(vals) + 0.18 * _avg(vals) + 0.07 * pair + 0.05 * tri


def build_memory(before_date: Optional[str], detail_csv: str) -> MemoryBank:
    b = MemoryBank()
    b.load_detail(detail_csv, before_date)
    b.load_memorybank(MEMORYBANK_CSV, before_date)
    return b


def build_memory5(before_date: Optional[str], detail_csv: str) -> MemoryBank5Plus:
    b = MemoryBank5Plus()
    b.load_detail(detail_csv, before_date)
    b.load_memorybank(MEMORYBANK_5PLUS_CSV, before_date)
    return b


def grade_bonus(ticket: Sequence[int], draws: Sequence[Draw]) -> float:
    t = tuple(sorted(ticket))
    low, mid, high = band_counts(t)
    odd = sum(n % 2 for n in t)
    total = sum(t)
    run = max_consecutive_run(t)
    repeat = len(set(t) & set(draws[-1].main))
    ldmax = max(Counter(n % 10 for n in t).values())
    score = 0.0
    if low in (2, 3) and mid in (2, 3) and high in (1, 2, 3):
        score += 0.30
    if odd in (3, 4):
        score += 0.22
    if 125 <= total <= 165:
        score += 0.25
    elif 115 <= total <= 175:
        score += 0.10
    if 2 <= repeat <= 4:
        score += 0.20
    if run <= 2:
        score += 0.15
    elif run >= 4:
        score -= 0.55
    if ldmax <= 2:
        score += 0.12
    elif ldmax >= 4:
        score -= 0.40
    return score


def third_prize_objective(matches: int, bonus_matches: int = 0) -> float:
    
    if matches >= 7:
        return 250.0
    if matches == 6:
        return 120.0 + min(bonus_matches, 2) * 15.0
    if matches == 5:
        return 28.0
    if matches == 4:
        return 7.0
    if matches == 3 and bonus_matches >= 1:
        return 2.0
    return max(0.0, matches - 1) * 0.25



def _recent_slice(draws: Sequence[Draw], window: int) -> Sequence[Draw]:
    return draws[-window:] if len(draws) > window else draws


def build_recent_context(draws: Sequence[Draw]) -> Dict[str, object]:
    """Gradi nezavisne modele na skorim prozorima (240/120/60).

    Walk-forward koristi samo prosli deo istorije — bez future leak-a.
    """
    ctx: Dict[str, object] = {}
    for window in (240, 120, 60):
        ds = list(_recent_slice(draws, window))
        ctx[f"num{window}"] = build_number_scores(ds)
        ctx[f"pair{window}"] = weighted_combination_counts(ds, 2, 55.0, 0)
        ctx[f"triple{window}"] = weighted_combination_counts(ds, 3, 45.0, 0)
    return ctx


def _norm_counter_value(counter: object, key: Tuple[int, ...]) -> float:
    try:
        return counter_norm(counter, key)  # type: ignore[arg-type]
    except Exception:
        return 0.0


def recent_window_score(ticket: Sequence[int], ctx: Dict[str, object]) -> float:
    t = tuple(sorted(ticket))
    vals: List[float] = []
    for window, weight in ((240, 0.45), (120, 0.35), (60, 0.20)):
        nums = ctx.get(f"num{window}", {})
        pairs = ctx.get(f"pair{window}", {})
        triples = ctx.get(f"triple{window}", {})
        if not isinstance(nums, dict):
            continue
        single = sum(float(nums.get(n, 0.0)) for n in t) / max(len(t), 1)
        pair = sum(_norm_counter_value(pairs, p) for p in itertools.combinations(t, 2)) / 21.0
        triple = sum(_norm_counter_value(triples, tri) for tri in itertools.combinations(t, 3)) / 35.0
        vals.append(weight * (0.58 * single + 0.30 * pair + 0.12 * triple))
    return sum(vals)


def pair_stability_score(ticket: Sequence[int], ctx: Dict[str, object]) -> float:
    """Parovi korisni u 240/120/60, ne samo u jednom skoku."""
    vals = []
    for p in itertools.combinations(tuple(sorted(ticket)), 2):
        v240 = _norm_counter_value(ctx.get("pair240", {}), p)
        v120 = _norm_counter_value(ctx.get("pair120", {}), p)
        v60 = _norm_counter_value(ctx.get("pair60", {}), p)
        mean = (v240 + v120 + v60) / 3.0
        spread = max(v240, v120, v60) - min(v240, v120, v60)
        vals.append(max(0.0, mean - 0.35 * spread))
    return _avg(vals)


def pair_recency_score(ticket: Sequence[int], ctx: Dict[str, object]) -> float:
    """Momentum parova sa naglaskom na prozore 60/120."""
    vals = []
    for p in itertools.combinations(tuple(sorted(ticket)), 2):
        v240 = _norm_counter_value(ctx.get("pair240", {}), p)
        v120 = _norm_counter_value(ctx.get("pair120", {}), p)
        v60 = _norm_counter_value(ctx.get("pair60", {}), p)
        vals.append(0.55 * v60 + 0.30 * v120 + 0.15 * v240)
    return _avg(vals)


def constraint_score(ticket: Sequence[int], draws: Sequence[Draw]) -> float:
    f = ticket_features(ticket, draws)
    base = float(f.get("constraint_score", 0.0))
    run = float(f.get("run", 0.0))
    digit_max = float(f.get("last_digit_max", 0.0))
    repeat_last = float(f.get("repeat_last", 0.0))
    penalty = 0.0
    if run >= 4:
        penalty += 0.30
    if digit_max >= 3:
        penalty += 0.18
    if repeat_last >= 4:
        penalty += 0.22
    return max(-0.5, base - penalty)


def ensemble_seed_score(ticket: Sequence[int], ctx: Dict[str, object]) -> float:
    """Score ako istu kombinaciju vole puni i skoriji pod-modeli."""
    t = tuple(sorted(ticket))
    score = 0.0
    for window in (240, 120, 60):
        pairs = ctx.get(f"pair{window}", {})
        nums = ctx.get(f"num{window}", {})
        if isinstance(nums, dict):
            score += sum(float(nums.get(n, 0.0)) for n in t) / max(len(t), 1) * (0.12 if window == 240 else 0.10)
        score += sum(_norm_counter_value(pairs, p) for p in itertools.combinations(t, 2)) / 21.0 * (0.18 if window == 60 else 0.12)
    return score

def context(draws: Sequence[Draw], before_date: Optional[str], detail_csv: str) -> Dict[str, object]:
    return {
        "num": build_number_scores(draws),
        "p1": weighted_combination_counts(draws, 2, 9.0, 70),
        "p2": weighted_combination_counts(draws, 2, 30.0, 180),
        "p3": weighted_combination_counts(draws, 2, 95.0, 0),
        "t1": weighted_combination_counts(draws, 3, 32.0, 220),
        "t2": weighted_combination_counts(draws, 3, 110.0, 0),
        "mem": build_memory(before_date, detail_csv),
        "mem5": build_memory5(before_date, detail_csv),
        "meta": load_meta_regressor(META_REGRESSOR_JSON),
        "clusters": load_cluster_profile(CLUSTER_CSV),
        "nextgen": load_nextgen_context(),
        "cycle_scores": build_cached_cycle_scores(draws),
        "cycle_gaps": build_cached_cycle_gaps(draws),
        **build_recent_context(draws),
    }


def build_cached_cycle_scores(draws: Sequence[Draw]) -> Dict[int, float]:
    try:
        from loto7_nextgen_models import cycle_number_scores
        return cycle_number_scores(draws)
    except Exception:
        return {}


def build_cached_cycle_gaps(draws: Sequence[Draw]) -> Dict[int, int]:
    try:
        from loto7_nextgen_models import last_seen_gaps
        return last_seen_gaps(draws)
    except Exception:
        return {}


def load_nextgen_context() -> Dict[str, object]:
    """Ucitava opcione nextgen JSON modele.

    Nedostajuci fajlovi = prazni modeli (radi i na cistom checkout-u).
    """
    try:
        from loto7_nextgen_models import load_meta6_classifier
    except Exception:
        load_meta6_classifier = None  # type: ignore
    ctx: Dict[str, object] = {"meta6": {}, "shap": {}}
    if load_meta6_classifier is not None:
        try:
            ctx["meta6"] = load_meta6_classifier(NEXTGEN_META6_JSON)
        except Exception:
            ctx["meta6"] = {}
    try:
        p = Path(NEXTGEN_SHAP_JSON)
        if p.exists():
            ctx["shap"] = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        ctx["shap"] = {}
    return ctx


def nextgen_scores(ticket: Sequence[int], draws: Sequence[Draw], ctx: Dict[str, object]) -> Dict[str, float]:
    try:
        from loto7_nextgen_models import meta6_score, selected_feature_score
    except Exception:
        return {"cycle": 0.0, "meta6": 0.0, "shap": 0.0}
    ng = ctx.get("nextgen", {}) if isinstance(ctx.get("nextgen", {}), dict) else {}
    t = tuple(sorted(ticket))
    # Cached cycle calculation: avoids recomputing per-number histories for each candidate.
    cycle_scores = ctx.get("cycle_scores", {}) if isinstance(ctx.get("cycle_scores", {}), dict) else {}
    cycle_gaps = ctx.get("cycle_gaps", {}) if isinstance(ctx.get("cycle_gaps", {}), dict) else {}
    if cycle_scores:
        base = sum(float(cycle_scores.get(n, 0.0)) for n in t) / max(len(t), 1)
        compat = 0.0
        pairs = 0
        for a, b in itertools.combinations(t, 2):
            compat += 1.0 / (1.0 + abs(float(cycle_gaps.get(a, 0)) - float(cycle_gaps.get(b, 0))) / 16.0)
            pairs += 1
        cycle = 0.72 * base + 0.28 * (compat / pairs if pairs else 0.0)
    else:
        cycle = 0.0
    try:
        meta6 = float(meta6_score(t, draws, ng.get("meta6", {})))  # type: ignore[union-attr]
    except Exception:
        meta6 = 0.0
    try:
        shap = float(selected_feature_score(t, draws, ng.get("shap", {})))  # type: ignore[union-attr]
    except Exception:
        shap = 0.0
    return {"cycle": cycle, "meta6": meta6, "shap": shap}


def cluster_key_from_features(f: Dict[str, float]) -> str:
    return f"L{int(f['low'])}_M{int(f['mid'])}_H{int(f['high'])}_O{int(f['odd'])}_R{int(f['run'])}"


def load_cluster_profile(path: str = CLUSTER_CSV) -> Dict[str, float]:
    p = Path(path)
    if not p.exists():
        return {}
    out: Dict[str, float] = {}
    try:
        for row in csv.DictReader(p.open("r", encoding="utf-8-sig", newline="")):
            key = row.get("cluster_key") or row.get("cluster") or ""
            if not key:
                continue
            out[key] = max(out.get(key, 0.0), float(row.get("score", 0) or 0))
    except Exception:
        return {}
    return out


def cluster_score(ticket: Sequence[int], draws: Sequence[Draw], profile: Dict[str, float]) -> float:
    if not profile:
        return 0.0
    f = ticket_features(ticket, draws)
    key = cluster_key_from_features(f)
    return float(profile.get(key, 0.0))


def build_hit_structure_clusters(detail_csv: str, output_csv: str = CLUSTER_CSV) -> Dict[str, float]:
    """Gradi jednostavnu tabelu klastera iz istorijskih visokih pogodaka."""
    p = Path(detail_csv)
    if not p.exists():
        return {}
    rows: List[Dict[str, object]] = []
    try:
        reader = csv.DictReader(p.open("r", encoding="utf-8-sig", newline=""))
        for row in reader:
            for i in range(1, 101):
                pk = f"predikcija{i}"
                mk = f"predikcija{i}_pogodaka"
                legacy_pk = f"予測{i}"
                if pk not in row and legacy_pk not in row:
                    break
                t = _parse_ticket(row.get(pk) or row.get(legacy_pk, ""))
                if not t:
                    continue
                try:
                    m = int(row.get(mk) or row.get(f"予測{i}_本数字一致", 0) or 0)
                except Exception:
                    m = 0
                if m < 4:
                    continue
                f = ticket_features(t, [])
                kolo = row.get("kolo") or row.get("抽せん日", "")
                rows.append({"kolo": kolo, "kombinacija": format_ticket(t), "pogodaka": m, **f})
    except Exception:
        return {}
    if not rows:
        return {}

    keys = ["sum", "odd", "low", "mid", "high", "run", "last_digit_max", "decade_count", "gap_avg", "gap_min", "gap_max"]
    labels: List[int]
    try:
        from sklearn.cluster import KMeans  # type: ignore

        X = [[float(r[k]) for k in keys] for r in rows]
        n_clusters = min(8, max(2, len(rows) // 20))
        labels = list(KMeans(n_clusters=n_clusters, random_state=42, n_init="auto").fit_predict(X))
    except Exception:
        labels = [int(r["low"]) * 100 + int(r["mid"]) * 10 + int(r["high"]) for r in rows]

    agg: Dict[str, List[float]] = defaultdict(list)
    for row, label in zip(rows, labels):
        ck = cluster_key_from_features({k: float(row[k]) for k in ticket_features((1, 2, 3, 4, 5, 6, 7), []).keys() if k in row})
        m = int(row.get("pogodaka") or row.get("一致数", 0) or 0)
        agg[ck].append(third_prize_objective(m, 0) / 120.0)
        row["cluster"] = label
        row["cluster_key"] = ck
        row["score"] = round(_avg(agg[ck]), 6)

    profile = {k: round(_avg(v), 6) for k, v in agg.items()}
    with Path(output_csv).open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = ["cluster_key", "score", "count"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for key, vals in sorted(agg.items(), key=lambda kv: (-_avg(kv[1]), kv[0])):
            w.writerow({"cluster_key": key, "score": round(_avg(vals), 6), "count": len(vals)})
    return profile


def load_meta_regressor(path: str = META_REGRESSOR_JSON) -> Dict[str, object]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def train_meta_regressor(detail_csv: str, output_json: str = META_REGRESSOR_JSON) -> Dict[str, object]:
    """Trenira ensemble regresora (RF + CatBoost + XGBoost) za ocekivani broj pogodaka."""
    p = Path(detail_csv)
    if not p.exists():
        return {}
    data: List[Tuple[Dict[str, float], float]] = []
    try:
        for row in csv.DictReader(p.open("r", encoding="utf-8-sig", newline="")):
            for i in range(1, 101):
                pk = f"predikcija{i}"
                mk = f"predikcija{i}_pogodaka"
                legacy_pk = f"予測{i}"
                if pk not in row and legacy_pk not in row:
                    break
                t = _parse_ticket(row.get(pk) or row.get(legacy_pk, ""))
                if not t:
                    continue
                try:
                    m = float(row.get(mk) or row.get(f"予測{i}_本数字一致", 0) or 0)
                except Exception:
                    m = 0.0
                data.append((ticket_features(t, []), m))
    except Exception:
        return {}
    if not data:
        return {}

    keys = list(data[0][0].keys())
    X = [[d[0][k] for k in keys] for d in data]
    y = [d[1] for d in data]

    # Linearni fallback (uvek dostupan u runtime-u)
    y_mean = sum(y) / len(y) if y else 0.0
    coefs, centers = [], []
    for j, key in enumerate(keys):
        xs = [row[j] for row in X]
        xm = sum(xs) / len(xs)
        num = sum((xs[i] - xm) * (y[i] - y_mean) for i in range(len(y)))
        den = sum((xs[i] - xm) ** 2 for i in range(len(y))) or 1.0
        coefs.append(num / den)
        centers.append(xm)

    info: Dict[str, object] = {
        "model": "linear_fallback",
        "seed": SEED,
        "total": len(y),
        "keys": keys,
        "coefs": coefs,
        "centers": centers,
        "regressors": {},
    }

    preds_sum = [0.0] * len(y)
    n_models = 0

    try:
        from sklearn.ensemble import RandomForestRegressor  # type: ignore

        rf = RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)
        rf.fit(X, y)
        pr = rf.predict(X)
        preds_sum = [preds_sum[i] + pr[i] for i in range(len(y))]
        n_models += 1
        info["regressors"]["random_forest"] = True
    except Exception:
        info["regressors"]["random_forest"] = False

    try:
        from catboost import CatBoostRegressor  # type: ignore

        cb = CatBoostRegressor(iterations=120, depth=4, learning_rate=0.06, verbose=False, random_seed=SEED)
        cb.fit(X, y)
        pr = cb.predict(X)
        preds_sum = [preds_sum[i] + float(pr[i]) for i in range(len(y))]
        n_models += 1
        info["regressors"]["catboost"] = True
    except Exception:
        info["regressors"]["catboost"] = False

    try:
        from xgboost import XGBRegressor  # type: ignore

        xgb = XGBRegressor(n_estimators=120, max_depth=4, learning_rate=0.06, random_state=SEED, verbosity=0)
        xgb.fit(X, y)
        pr = xgb.predict(X)
        preds_sum = [preds_sum[i] + float(pr[i]) for i in range(len(y))]
        n_models += 1
        info["regressors"]["xgboost"] = True
    except Exception:
        info["regressors"]["xgboost"] = False

    if n_models > 0:
        ens = [preds_sum[i] / n_models for i in range(len(y))]
        # Podesi linear fallback koef. prema ensemble-u
        for j in range(len(keys)):
            xs = [row[j] for row in X]
            xm = sum(xs) / len(xs)
            em = sum(ens) / len(ens)
            num = sum((xs[i] - xm) * (ens[i] - em) for i in range(len(ens)))
            den = sum((xs[i] - xm) ** 2 for i in range(len(xs))) or 1.0
            coefs[j] = num / den
            centers[j] = xm
        info["model"] = f"ensemble_{n_models}_regressors_json_fallback"
        info["coefs"] = coefs
        info["centers"] = centers

    Path(output_json).write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return info


def meta_score(ticket: Sequence[int], draws: Sequence[Draw], meta: Dict[str, object]) -> float:
    if not meta:
        return 0.0
    keys = meta.get("keys", [])
    coefs = meta.get("coefs", [])
    centers = meta.get("centers", [])
    if not isinstance(keys, list) or not isinstance(coefs, list):
        return 0.0
    f = ticket_features(ticket, draws)
    raw = 0.0
    for i, key in enumerate(keys):
        try:
            center = float(centers[i]) if isinstance(centers, list) and i < len(centers) else 0.0
            raw += (float(f.get(key, 0.0)) - center) * float(coefs[i])
        except Exception:
            continue
    # bounded logistic-like score
    return max(-0.6, min(0.6, raw / 120.0))


def score_ticket(ticket: Sequence[int], draws: Sequence[Draw], ctx: Dict[str, object], w: AdvancedWeights, strategy: str) -> TicketScore:
    t = tuple(sorted(ticket))
    nums: Dict[int, float] = ctx["num"]  # type: ignore
    single = sum(nums.get(n, 0.0) for n in t)
    pair = sum(
        0.40 * counter_norm(ctx["p1"], p) + 0.28 * counter_norm(ctx["p2"], p) + 0.10 * counter_norm(ctx["p3"], p)
        for p in itertools.combinations(t, 2)
    )  # type: ignore
    triple = sum(
        0.52 * counter_norm(ctx["t1"], tri) + 0.18 * counter_norm(ctx["t2"], tri)
        for tri in itertools.combinations(t, 3)
    )  # type: ignore
    mem: MemoryBank = ctx["mem"]  # type: ignore
    mem5: MemoryBank5Plus = ctx["mem5"]  # type: ignore
    memory = mem.score(t)
    memory5 = mem5.score(t)
    gb = grade_bonus(t, draws)
    penalty = structure_penalty(t, draws)
    low, mid, high = band_counts(t)
    div = 0.10 if len(set(n // 10 for n in t)) >= 4 else 0.0
    ms = meta_score(t, draws, ctx.get("meta", {}))  # type: ignore[arg-type]
    cs = cluster_score(t, draws, ctx.get("clusters", {}))  # type: ignore[arg-type]
    ngs = nextgen_scores(t, draws, ctx)
    cycle = ngs.get("cycle", 0.0)
    meta6 = ngs.get("meta6", 0.0)
    shap = ngs.get("shap", 0.0)
    recent = recent_window_score(t, ctx)
    stability = pair_stability_score(t, ctx)
    recency = pair_recency_score(t, ctx)
    constraint = constraint_score(t, draws)
    ensemble = ensemble_seed_score(t, ctx)
    bonus = 0.45 if strategy == "GRADE3" and (memory > 0 or memory5 > 0) else 0.22 if strategy == "DIFFUSION" else 0.20 if strategy == "MULTI_AGENT_PPO" else 0.18 if strategy.startswith("RECENT") else 0.15 if strategy == "MCTS" else 0.08 if strategy == "MONTECARLO" else 0.0
    total = (
        w.single * single
        + w.pair * pair
        + w.triple * triple
        + w.memory * memory
        + w.memory5 * memory5
        + w.grade6 * gb
        + w.diversity * div
        + w.meta * ms
        + w.cluster * cs
        + w.cycle * cycle
        + w.meta6 * meta6
        + w.shap * shap
        + w.recent * recent
        + w.pair_stability * stability
        + w.pair_recency * recency
        + w.constraint * constraint
        + w.ensemble * ensemble
        + bonus
        - w.structure * penalty
    )
    return TicketScore(
        t,
        total,
        {
            "single": single,
            "pair": pair,
            "triple": triple,
            "memory": memory,
            "memory5": memory5,
            "pattern": memory,
            "grade6": gb,
            "meta": ms,
            "cluster": cs,
            "cycle": cycle,
            "meta6": meta6,
            "shap": shap,
            "recent": recent,
            "pair_stability": stability,
            "pair_recency": recency,
            "constraint": constraint,
            "ensemble": ensemble,
            "penalty": penalty,
            "sum": float(sum(t)),
            "odd": float(sum(n % 2 for n in t)),
            "low": float(low),
            "mid": float(mid),
            "high": float(high),
            "repeat_last": float(len(set(t) & set(draws[-1].main))),
        },
        strategy,
    )


def sample_ticket(draws: Sequence[Draw], pool_size: int, rng: random.Random) -> Tuple[int, ...]:
    ns = build_number_scores(draws)
    pool = make_candidate_pool(draws, max(pool_size, 16))
    alln = list(range(1, 40))
    hot = sorted(alln, key=lambda n: ns.get(n, 0), reverse=True)[: max(20, pool_size)]
    s = set()
    while len(s) < 7:
        r = rng.random()
        s.add(rng.choice(pool if r < 0.70 else hot if r < 0.90 else alln))
    return tuple(sorted(s))


def mcts_candidates(
    draws: Sequence[Draw],
    ctx: Dict[str, object],
    weights: AdvancedWeights,
    pool_size: int,
    iterations: int,
    seed: int,
) -> List[TicketScore]:
    """Lagani UCB MCTS pri dodavanju brojeva."""
    if iterations <= 0:
        return []
    rng = random.Random(seed)
    ns = build_number_scores(draws)
    pool = make_candidate_pool(draws, max(pool_size, 18))
    alln = list(range(1, 40))
    visits: Counter = Counter()
    reward: Counter = Counter()
    best: Dict[Tuple[int, ...], TicketScore] = {}

    def choose_number(current: set[int]) -> int:
        choices = pool if rng.random() < 0.75 else alln
        sample = [n for n in rng.sample(choices, min(len(choices), 14)) if n not in current]
        if not sample:
            sample = [n for n in alln if n not in current]
        total_visits = sum(visits.values()) + 2
        best_n = sample[0]
        best_v = -10**9
        for n in sample:
            avg = reward[n] / (visits[n] + 1)
            ucb = avg + 0.35 * math.sqrt(math.log(total_visits) / (visits[n] + 1)) + ns.get(n, 0.0) * 0.025
            if ucb > best_v:
                best_n, best_v = n, ucb
        return best_n

    for _ in range(iterations):
        cur: set[int] = set()
        while len(cur) < PICK_SIZE:
            cur.add(choose_number(cur))
        ticket = tuple(sorted(cur))
        scored = score_ticket(ticket, draws, ctx, weights, "MCTS")
        # MCTS backpropagation uses the ranking score plus 5+ memory/meta.
        r = scored.score + scored.detail.get("memory5", 0.0) * 2.0 + scored.detail.get("meta", 0.0)
        for n in ticket:
            visits[n] += 1
            reward[n] += r
        old = best.get(ticket)
        if old is None or scored.score > old.score:
            best[ticket] = scored
    return sorted(best.values(), key=lambda x: (-x.score, x.ticket))[: max(500, min(iterations, 2000))]


def optimize_weights(draws: Sequence[Draw], trials: int = 25, force: bool = False) -> AdvancedWeights:
    cached = None if force else _load_weights()
    if cached:
        return cached
    rng = random.Random(SEED)
    best = AdvancedWeights()
    best_score = -1.0

    def eval_w(w: AdvancedWeights) -> float:
        start = max(20, len(draws) - 36)
        val = []
        for i in range(start, len(draws)):
            preds = advanced_predict(
                draws[:i],
                5,
                15,
                DEFAULT_HIT_PATTERN_CSV,
                draws[i].date,
                200,
                300,
                w,
                False,
            )
            scores = []
            for p in preds:
                m = count_main_matches(p.ticket, draws[i].main)
                scores.append(third_prize_objective(m, 0))
            val.append(max(scores) if scores else 0.0)
        return _avg(val)

    try:
        import optuna  # type: ignore

        def obj(trial):
            w = AdvancedWeights(
                single=trial.suggest_float("single", 0.55, 1.45),
                pair=trial.suggest_float("pair", 0.65, 1.85),
                triple=trial.suggest_float("triple", 0.75, 2.10),
                memory=trial.suggest_float("memory", 0.60, 2.10),
                grade6=trial.suggest_float("grade6", 1.00, 3.20),
                structure=trial.suggest_float("structure", 0.55, 1.75),
                diversity=trial.suggest_float("diversity", 0.02, 0.70),
                memory5=trial.suggest_float("memory5", 0.90, 3.50),
                meta=trial.suggest_float("meta", 0.05, 1.20),
                cluster=trial.suggest_float("cluster", 0.00, 0.80),
                cycle=trial.suggest_float("cycle", 0.10, 1.20),
                diffusion=trial.suggest_float("diffusion", 0.05, 0.90),
                ppo=trial.suggest_float("ppo", 0.05, 0.90),
                meta6=trial.suggest_float("meta6", 0.10, 1.80),
                shap=trial.suggest_float("shap", 0.00, 0.80),
                recent=trial.suggest_float("recent", 0.20, 1.60),
                pair_stability=trial.suggest_float("pair_stability", 0.10, 1.30),
                pair_recency=trial.suggest_float("pair_recency", 0.10, 1.40),
                constraint=trial.suggest_float("constraint", 0.25, 1.70),
                ensemble=trial.suggest_float("ensemble", 0.05, 1.20),
            )
            return eval_w(w)

        st = optuna.create_study(direction="maximize")
        st.optimize(obj, n_trials=int(os.getenv("LOTO7_OPTUNA_TRIALS", str(trials))), show_progress_bar=False)
        best = AdvancedWeights.from_dict(st.best_params)
        best_score = float(st.best_value)
    except Exception:
        for _ in range(int(os.getenv("LOTO7_OPTUNA_TRIALS", str(trials)))):
            w = AdvancedWeights(
                single=rng.uniform(0.55, 1.45),
                pair=rng.uniform(0.65, 1.85),
                triple=rng.uniform(0.75, 2.10),
                memory=rng.uniform(0.60, 2.10),
                grade6=rng.uniform(1.00, 3.20),
                structure=rng.uniform(0.55, 1.75),
                diversity=rng.uniform(0.02, 0.70),
                memory5=rng.uniform(0.90, 3.50),
                meta=rng.uniform(0.05, 1.20),
                cluster=rng.uniform(0.00, 0.80),
                cycle=rng.uniform(0.10, 1.20),
                diffusion=rng.uniform(0.05, 0.90),
                ppo=rng.uniform(0.05, 0.90),
                meta6=rng.uniform(0.10, 1.80),
                shap=rng.uniform(0.00, 0.80),
                recent=rng.uniform(0.20, 1.60),
                pair_stability=rng.uniform(0.10, 1.30),
                pair_recency=rng.uniform(0.10, 1.40),
                constraint=rng.uniform(0.25, 1.70),
                ensemble=rng.uniform(0.05, 1.20),
            )
            sc = eval_w(w)
            if sc > best_score:
                best, best_score = w, sc
    _save_weights(best)
    return best


def advanced_predict(
    draws: Sequence[Draw],
    num_tickets: int = 10,
    pool_size: int = 24,
    hit_pattern_csv: str = DEFAULT_HIT_PATTERN_CSV,
    before_date: Optional[str] = None,
    monte_carlo_iterations: Optional[int] = None,
    mcts_iterations: Optional[int] = None,
    weights: Optional[AdvancedWeights] = None,
    optimize: bool = True,
) -> List[TicketScore]:
    if weights is None:
        weights = optimize_weights(draws) if optimize and len(draws) >= 120 and os.getenv("LOTO7_DISABLE_OPTIMIZE", "0") != "1" else (_load_weights() or AdvancedWeights())
    mc = int(os.getenv("LOTO7_MONTE_CARLO", str(monte_carlo_iterations if monte_carlo_iterations is not None else 12000)))
    mcts = int(os.getenv("LOTO7_MCTS_ITERATIONS", str(mcts_iterations if mcts_iterations is not None else 5000)))
    ctx = context(draws, before_date, hit_pattern_csv)
    ranked: Dict[Tuple[int, ...], TicketScore] = {}
    pool = make_candidate_pool(draws, pool_size)

    for t in itertools.combinations(pool, 7):
        item = score_ticket(t, draws, ctx, weights, "GRADE3")
        ranked[item.ticket] = item

    # Ensemble candidate generation from recent-window sub-models.
    if os.getenv("LOTO7_SKIP_RECENT", "0") != "1":
        for window in (240, 120, 60):
            if len(draws) < max(30, window // 2):
                continue
            recent_draws = list(_recent_slice(draws, window))
            recent_pool_cap = int(os.getenv("LOTO7_RECENT_POOL_SIZE", "17"))
            default_recent_pool = 18 if window == 240 else 17 if window == 120 else 16
            recent_pool = make_candidate_pool(recent_draws, min(NUM_MAX, max(pool_size, min(default_recent_pool, recent_pool_cap))))
            for t in itertools.combinations(recent_pool, 7):
                item = score_ticket(t, draws, ctx, weights, f"RECENT{window}")
                item.score += weights.ensemble * (0.18 if window == 60 else 0.12)
                old = ranked.get(item.ticket)
                if old is None or item.score > old.score:
                    ranked[item.ticket] = item

    rng = random.Random(len(draws) * 1009 + sum(ord(c) for c in draws[-1].date))
    for _ in range(max(0, mc)):
        item = score_ticket(sample_ticket(draws, pool_size, rng), draws, ctx, weights, "MONTECARLO")
        old = ranked.get(item.ticket)
        if old is None or item.score > old.score:
            ranked[item.ticket] = item

    for item in mcts_candidates(draws, ctx, weights, pool_size, max(0, mcts), seed=len(draws) * 777):
        old = ranked.get(item.ticket)
        if old is None or item.score > old.score:
            ranked[item.ticket] = item

    if os.getenv("LOTO7_DISABLE_NEXTGEN", "0") != "1":
        try:
            from loto7_nextgen_models import diffusion_candidates, multi_agent_ppo_candidates
            diff_count = int(os.getenv("LOTO7_DIFFUSION_CANDIDATES", "1200"))
            ppo_count = int(os.getenv("LOTO7_PPO_CANDIDATES", "1000"))
            for t in diffusion_candidates(draws, diff_count, seed=len(draws) * 1777):
                item = score_ticket(t, draws, ctx, weights, "DIFFUSION")
                item.score += weights.diffusion * item.detail.get("cycle", 0.0)
                old = ranked.get(item.ticket)
                if old is None or item.score > old.score:
                    ranked[item.ticket] = item
            for t in multi_agent_ppo_candidates(draws, ppo_count, seed=len(draws) * 2777):
                item = score_ticket(t, draws, ctx, weights, "MULTI_AGENT_PPO")
                item.score += weights.ppo * (item.detail.get("cycle", 0.0) + item.detail.get("meta6", 0.0))
                old = ranked.get(item.ticket)
                if old is None or item.score > old.score:
                    ranked[item.ticket] = item
        except Exception as exc:
            if os.getenv("LOTO7_DEBUG_NEXTGEN", "0") == "1":
                print(f"[UPOZ] nextgen generisanje kandidata preskoceno: {exc}")

    selected: List[TicketScore] = []
    use: Counter = Counter()
    for item in sorted(ranked.values(), key=lambda x: (-x.score, x.ticket)):
        if any(len(set(item.ticket) & set(s.ticket)) > 4 for s in selected):
            continue
        if any(use[n] >= 3 for n in item.ticket):
            continue
        selected.append(item)
        use.update(item.ticket)
        if len(selected) >= num_tickets:
            return selected
    return sorted(ranked.values(), key=lambda x: (-x.score, x.ticket))[:num_tickets]


def _write_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _load_existing_detail(detail_csv: str) -> Tuple[List[Dict[str, object]], set[str]]:
    p = Path(detail_csv)
    if not p.exists():
        return [], set()
    try:
        rows = list(csv.DictReader(p.open("r", encoding="utf-8-sig", newline="")))
        done = {str(r.get("kolo") or r.get("抽せん日", "")) for r in rows if r.get("kolo") or r.get("抽せん日")}
        return rows, done
    except Exception:
        return [], set()


def _save_resume(path: str, completed: int, last_date: str) -> None:
    Path(path).write_text(json.dumps({"completed": completed, "last_date": last_date}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _maybe_git_push(paths: Sequence[str], message: str) -> None:
    """Opcioni git push za GitHub Actions (podrazumevano iskljucen: LOTO7_ENABLE_AUTO_PUSH=1)."""
    if os.getenv("LOTO7_ENABLE_AUTO_PUSH", "0") != "1":
        return
    try:
        subprocess.run(["git", "config", "user.name", "github-actions"], check=False)
        subprocess.run(["git", "config", "user.email", "github-actions@github.com"], check=False)
        subprocess.run(["git", "add", *paths], check=False)
        diff = subprocess.run(["git", "diff", "--cached", "--quiet"], check=False)
        if diff.returncode != 0:
            subprocess.run(["git", "commit", "-m", message], check=False)
            subprocess.run(["git", "push"], check=False)
    except Exception:
        return


def summarize_rows(rows: Sequence[Dict[str, object]], num_tickets: int, min_train: int, start_index: int, weights: AdvancedWeights) -> Dict[str, object]:
    best = [int(r.get("max_pogodaka") or r.get("最高本数字一致数", 0) or 0) for r in rows]
    top = []
    for row in rows:
        try:
            top.append(int(row.get("predikcija1_pogodaka") or row.get("予測1_本数字一致", 0) or 0))
        except Exception:
            top.append(0)
    purchase = 0
    prize = 0

    def rate(v: Sequence[int], t: int) -> float:
        return sum(1 for x in v if x >= t) / len(v) if v else 0.0

    return {
        "broj_validacija": len(rows),
        "pocetna_obuka": min_train,
        "pocetak_validacije": start_index + 1,
        "komb_po_kolu": num_tickets,
        "prosek_pog_prva": round(_avg(top), 6),
        "prosek_pog_max": round(_avg(best), 6),
        "stopa_prva_3plus": round(rate(top, 3), 6),
        "stopa_prva_4plus": round(rate(top, 4), 6),
        "stopa_prva_5plus": round(rate(top, 5), 6),
        "stopa_max_3plus": round(rate(best, 3), 6),
        "stopa_max_4plus": round(rate(best, 4), 6),
        "stopa_max_5plus": round(rate(best, 5), 6),
        "stopa_max_6plus": round(rate(best, 6), 6),
        "distribucija_prva": dict(sorted(Counter(top).items())),
        "distribucija_max": dict(sorted(Counter(best).items())),
        "tezine": weights.to_dict(),
    }


def advanced_backtest(
    draws: Sequence[Draw],
    min_train: int,
    num_tickets: int,
    pool_size: int,
    hit_pattern_csv: str,
    max_backtest_draws: int,
    summary_csv: str,
    detail_csv: str,
) -> Dict[str, object]:
    start = max(min_train, len(draws) - max_backtest_draws) if max_backtest_draws and max_backtest_draws > 0 else min_train
    weights = (
        optimize_weights(draws[:start])
        if start >= 120 and os.getenv("LOTO7_DISABLE_OPTIMIZE", "0") != "1"
        else (_load_weights() or AdvancedWeights())
    )
    resume_enabled = os.getenv("LOTO7_BACKTEST_RESUME", "1") != "0"
    push_every = int(os.getenv("LOTO7_PUSH_EVERY", "100"))
    mcts_backtest = int(os.getenv("LOTO7_BACKTEST_MCTS", os.getenv("LOTO7_MCTS_ITERATIONS", "1500")))

    rows, done = _load_existing_detail(detail_csv) if resume_enabled else ([], set())
    bank = MemoryBank()
    bank4 = MemoryBank()
    bank6 = MemoryBank5Plus()
    bank5 = MemoryBank5Plus()

    processed_since_save = 0
    for i in range(start, len(draws)):
        actual = draws[i]
        if resume_enabled and actual.date in done:
            continue
        preds = advanced_predict(
            draws[:i],
            num_tickets,
            pool_size,
            hit_pattern_csv,
            actual.date,
            int(os.getenv("LOTO7_BACKTEST_MONTE_CARLO", "2500")),
            mcts_backtest,
            weights,
            False,
        )
        hits = [count_main_matches(p.ticket, actual.main) for p in preds]
        row: Dict[str, object] = {
            "kolo": actual.date,
            "broj_kola": actual.draw_no or "",
            "izvuceno": format_ticket(actual.main),
            "broj_komb": len(preds),
            "max_pogodaka": max(hits) if hits else 0,
        }
        for idx, (p, h) in enumerate(zip(preds, hits), 1):
            row[f"predikcija{idx}"] = format_ticket(p.ticket)
            row[f"predikcija{idx}_strategija"] = p.strategy
            row[f"predikcija{idx}_pogodaka"] = h
            if h >= 4:
                bank.add(p.ticket, 1.0 + max(0, h - 4) * 0.8)
                bank4.add(p.ticket, 1.0 + max(0, h - 4) * 0.8)
            if h >= 5:
                bank5.add(p.ticket, h, actual.date)
            if h >= 6:
                bank6.add(p.ticket, h, actual.date)
        rows.append(row)
        done.add(actual.date)
        processed_since_save += 1

        if push_every > 0 and processed_since_save >= push_every:
            _write_csv(detail_csv, rows)
            _save_resume(RESUME_JSON, len(done), actual.date)
            bank.save(MEMORYBANK_CSV)
            bank4.save(MEMORYBANK_4PLUS_CSV)
            bank5.save(MEMORYBANK_5PLUS_CSV)
            bank6.save(MEMORYBANK_6HIT_CSV)
            partial_summary = summarize_rows(rows, num_tickets, min_train, start, weights)
            _write_csv(summary_csv, [partial_summary])
            _maybe_git_push(
                [detail_csv, summary_csv, MEMORYBANK_CSV, MEMORYBANK_4PLUS_CSV, MEMORYBANK_5PLUS_CSV, MEMORYBANK_6HIT_CSV, RESUME_JSON],
                f"checkpoint loto7 backtest {len(done)} draws",
            )
            processed_since_save = 0

    _write_csv(detail_csv, rows)
    summary = summarize_rows(rows, num_tickets, min_train, start, weights)
    _write_csv(summary_csv, [summary])
    bank.save(MEMORYBANK_CSV)
    bank4.save(MEMORYBANK_4PLUS_CSV)
    bank5.save(MEMORYBANK_5PLUS_CSV)
    bank6.save(MEMORYBANK_6HIT_CSV)
    _save_resume(RESUME_JSON, len(done), rows[-1].get("kolo") or rows[-1].get("抽せん日", "") if rows else "")
    build_hit_structure_clusters(detail_csv, CLUSTER_CSV)
    train_meta_regressor(detail_csv, META_REGRESSOR_JSON)
    try:
        from loto7_nextgen_models import train_meta6_classifier, shap_feature_selection
        train_meta6_classifier(detail_csv, NEXTGEN_META6_JSON)
        shap_feature_selection(detail_csv, NEXTGEN_SHAP_JSON)
    except Exception as exc:
        if os.getenv("LOTO7_DEBUG_NEXTGEN", "0") == "1":
            print(f"[UPOZ] nextgen obuka preskocena: {exc}")
    return summary


__all__ = [
    "AdvancedWeights",
    "MemoryBank",
    "MemoryBank5Plus",
    "MEMORYBANK_4PLUS_CSV",
    "MEMORYBANK_6HIT_CSV",
    "build_recent_context",
    "recent_window_score",
    "pair_stability_score",
    "pair_recency_score",
    "advanced_predict",
    "advanced_backtest",
    "optimize_weights",
    "third_prize_objective",
    "train_meta_regressor",
    "build_hit_structure_clusters",
    "load_nextgen_context",
    "nextgen_scores",
    "save_latest_txt",
]

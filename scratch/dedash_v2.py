"""Consistency de-dash of paper/main.tex: standardize prose compounds to the unhyphenated
style used in the body, so the Mac-written sections (related work, captions, appendix) match.

PROTECTS: the bibliography (page ranges + verbatim titles), proper/test names (Engle-Granger,
Newey-West, Dickey-Fuller, Ljung-Box, Ornstein-Uhlenbeck, etc.), and cross-reference label keys
(kalman-placebo, ofi-decay, circuit-breaker, ...), because none of those are in the curated dict.
Run: python scratch/dedash_v2.py
"""
import re

P = r"C:\Users\jackw\Desktop\math-199-research\paper\main.tex"
src = open(P, encoding="utf-8").read()
i = src.index(r"\begin{thebibliography}")
body, bib = src[:i], src[i:]   # bib left untouched

PAIRS = {
    "out-of-sample": "out of sample", "Out-of-sample": "Out of sample",
    "in-sample": "in sample", "In-sample": "In sample",
    "market-neutral": "market neutral", "mean-reverting": "mean reverting",
    "mean-reversion": "mean reversion", "no-stop": "no stop", "No-stop": "No stop",
    "walk-forward": "walk forward", "Walk-forward": "Walk forward",
    "held-out": "held out", "order-flow": "order flow", "Order-flow": "Order flow",
    "point-in-time": "point in time", "trade-only": "trade only",
    "cross-exchange": "cross exchange", "high-frequency": "high frequency",
    "high-volatility": "high volatility", "structural-break": "structural break",
    "perfect-foresight": "perfect foresight", "serial-correlation": "serial correlation",
    "rolling-mean": "rolling mean", "best-level": "best level", "event-time": "event time",
    "per-order": "per order", "per-leg": "per leg", "per-trade": "per trade",
    "size-weighted": "size weighted", "steady-state": "steady state",
    "most-traded": "most traded", "matched-placebo": "matched placebo",
    "weak-form": "weak form", "gradient-boosted": "gradient boosted",
    "random-walk": "random walk", "Random-walk": "Random walk",
    "phase-randomized": "phase randomized", "block-shuffled": "block shuffled",
    "error-correction": "error correction", "time-varying": "time varying",
    "two-step": "two step", "one-step": "one step", "four-hour": "four hour",
    "size-independent": "size independent", "log-price": "log price",
    "cross-validation": "cross validation", "real-time": "real time",
    "closed-form": "closed form", "co-movement": "comovement", "co-moves": "comoves",
    "multi-week": "multiweek", "multi-year": "multiyear",
    "non-anticipative": "nonanticipative", "non-negative": "nonnegative",
    "non-linear": "nonlinear", "non-overlapping": "nonoverlapping", "non-zero": "nonzero",
    "well-specified": "well specified", "cost-sensitive": "cost sensitive",
    "HAC-adjusted": "HAC adjusted", "Coinbase-adjusted": "Coinbase adjusted",
    "venue-centered": "venue centered", "per-pair": "per pair", "Per-pair": "Per pair",
    "per-symbol": "per symbol", "Per-symbol": "Per symbol", "per-event": "per event",
    "per-split": "per split", "signal-timed": "signal timed", "Signal-timed": "Signal timed",
    "bar-chart": "bar chart", "capital-control": "capital control",
    "deep-learning": "deep learning", "discrete-time": "discrete time",
    "distance-based": "distance based", "event-level": "event level",
    "half-life": "half life", "honest-accounting": "honest accounting",
    "limit-order": "limit order", "liquid-major": "liquid major", "log-spread": "log spread",
    "market-efficiency": "market efficiency", "multiple-testing": "multiple testing",
    "pair-screening": "pair screening", "pairs-trading": "pairs trading",
    "pass-rate": "pass rate", "practitioner-facing": "practitioner facing",
    "random-pair": "random pair", "rank-shuffled": "rank shuffled",
    "seconds-scale": "seconds scale", "single-leg": "single leg",
    "state-space": "state space", "sub-second": "subsecond",
    "permanent-versus": "permanent versus", "four-architecture": "four architecture",
    "order-book": "order book", "Per-trade": "Per trade", "versus-transient": "versus transient",
}
for k in sorted(PAIRS, key=len, reverse=True):
    body = re.sub(r"(?<![A-Za-z])" + re.escape(k) + r"(?![A-Za-z])", PAIRS[k], body)

# $x$-notation: $z$-score -> $z$ score, $t$-ratio -> $t$ ratio
body = re.sub(r"(\$[a-zA-Z]\$)-(\w)", r"\1 \2", body)
# numeric en-dash ranges in prose -> "to"
body = re.sub(r"(\$?[+-]?\d[\d.,]*\\?%?\$?)--(\$?[+-]?\d[\d.,]*\\?%?\$?)", r"\1 to \2", body)

open(P, "w", encoding="utf-8", newline="\n").write(body + bib)

print("=== remaining hyphenated compounds in body (proper names + label keys should be all that's left) ===")
for h in sorted(set(re.findall(r"[A-Za-z]{2,}-[A-Za-z]{2,}", body))):
    print("  ", h)

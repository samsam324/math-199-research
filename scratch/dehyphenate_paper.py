"""
One-off: strip most dashes from paper/main.tex prose (Jack wants it to read like an undergrad,
not AI). Protects the bibliography (page ranges, proper titles), citation keys, and cross-reference
labels so nothing breaks. Reports any dashes left for manual review.

Run: python scratch/dehyphenate_paper.py
"""
import os, re

P = r"C:\Users\jackw\Desktop\math-199-research\paper\main.tex"
src = open(P, encoding="utf-8").read()

# Protect the bibliography wholesale (page ranges + proper titles keep their dashes).
i = src.index(r"\begin{thebibliography}")
body, bib = src[:i], src[i:]

# Curated compound -> de-hyphenated. Longer keys first so e.g. micro-caps beats micro-cap.
PAIRS = {
    "out-of-sample": "out of sample", "statistical-arbitrage": "statistical arbitrage",
    "backtest-overfitting": "backtest overfitting", "survivorship-fragile": "survivorship fragile",
    "selection-sensitive": "selection sensitive", "phase-randomized": "phase randomized",
    "pre-registration": "preregistration", "pre-registered": "preregistered",
    "window-dependent": "window dependent", "venue-adjusted": "venue adjusted",
    "venue-honest": "venue honest", "block-shuffled": "block shuffled",
    "mean-reversion": "mean reversion", "mean-reverting": "mean reverting",
    "market-neutral": "market neutral", "random-walk": "random walk", "static-OLS": "static OLS",
    "dynamic-hedge": "dynamic hedge", "deflated-Sharpe": "deflated Sharpe",
    "stop-sensitive": "stop sensitive", "pairs-trading": "pairs trading",
    "error-correction": "error correction", "high-volatility": "high volatility",
    "time-varying": "time varying", "unit-notional": "unit notional", "trade-only": "trade only",
    "train-window": "train window", "test-window": "test window", "test-period": "test period",
    "white-noise": "white noise", "weak-form": "weak form", "USDT-quoted": "USDT quoted",
    "USD-quoted": "USD quoted", "event-level": "event level", "order-flow": "order flow",
    "point-in-time": "point in time", "equal-weight": "equal weight", "long-horizon": "long horizon",
    "long-short": "long short", "best-quote": "best quote", "leg-crash": "leg crash",
    "risk-off": "risk off", "held-out": "held out", "in-sample": "in sample",
    "never-stop": "never stop", "no-stop": "no stop", "real-time": "real time",
    "first-pass": "first pass", "two-leg": "two leg", "one-second": "one second",
    "four-hour": "four hour", "three-month": "three month", "six-month": "six month",
    "walk-forward": "walk forward", "half-life": "half life", "co-movement": "comovement",
    "non-circular": "noncircular", "non-committal": "noncommittal", "mid-price": "midprice",
    "multi-year": "multiyear", "multi-week": "multiweek", "micro-caps": "microcaps",
    "micro-cap": "microcap", "near-white": "nearly white", "top-50": "top 50",
}
for k in sorted(PAIRS, key=len, reverse=True):
    body = re.sub(r"(?<![A-Za-z])" + re.escape(k) + r"(?![A-Za-z])", PAIRS[k], body)

# Notation: $z$-score -> $z$ score, $t$-statistic -> $t$ statistic, rolling-$z$ -> rolling $z$.
body = re.sub(r"(\$[a-zA-Z]\$)-(\w)", r"\1 \2", body)
body = re.sub(r"(\w)-(\$[a-zA-Z]\$)", r"\1 \2", body)

# number-unit: 35-day -> 35 day, 30-second -> 30 second, etc.
body = re.sub(r"(\d)-(day|days|hour|hours|month|months|week|weeks|year|years|"
              r"pair|pairs|bps|leg|legs|sample|bar|bars|second|seconds|state|column)\b",
              r"\1 \2", body)

# named tests (en-dash) -> spelled out.
body = body.replace("Newey--West", "Newey and West").replace("Engle--Granger", "Engle and Granger")

# "near-$100\%$" -> "nearly $100\%$"
body = body.replace(r"near-$100\%$", r"nearly $100\%$").replace("near-100", "nearly 100")

# the one em-dash: a "not applicable" table cell.
body = body.replace("& --- \\\\", "& n/a \\\\")

# numeric en-dash ranges in prose -> "to" (years and value ranges).
body = re.sub(r"(\$?[+-]?\d[\d.,]*\\?%?\$?)--(\$?[+-]?\d[\d.,]*\\?%?\$?)", r"\1 to \2", body)

src2 = body + bib
open(P, "w", encoding="utf-8", newline="\n").write(src2)

# Report what dashes remain in the BODY for manual review.
def show(label, pattern, text):
    hits = re.findall(pattern, text)
    print(f"{label}: {len(hits)}")
    for h in sorted(set(hits)):
        print("   ", h if isinstance(h, str) else " ".join(h))

print("=== remaining in body (review these) ===")
show("hyphen compounds (word-word)", r"[A-Za-z]{2,}-[A-Za-z]{2,}", body)
show("en-dashes --", r"--", body)
show("em-dashes ---", r"---", body)
print("\n(bibliography left untouched on purpose: page ranges + proper titles)")

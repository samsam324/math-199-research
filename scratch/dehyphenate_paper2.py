"""Second de-hyphenation pass: the compounds the first pass missed. Leaves the two label keys
(kalman-placebo, kalman-freq) and the bibliography alone. Run: python scratch/dehyphenate_paper2.py"""
import re

P = r"C:\Users\jackw\Desktop\math-199-research\paper\main.tex"
src = open(P, encoding="utf-8").read()
i = src.index(r"\begin{thebibliography}")
body, bib = src[:i], src[i:]

PAIRS = {
    "stop-versus-no-stop": "stop versus no stop", "train-to-test": "train to test",
    "Out-of-sample": "Out of sample", "non-overlapping": "nonoverlapping",
    "perfect-foresight": "perfect foresight", "frozen-parameter": "frozen parameter",
    "principal-component": "principal component", "structural-break": "structural break",
    "multiple-testing": "multiple testing", "machine-learning": "machine learning",
    "data-contingent": "data contingent", "size-independent": "size independent",
    "delisting-scale": "delisting scale", "iceberg-episode": "iceberg episode",
    "longer-horizon": "longer horizon", "seconds-scale": "seconds scale",
    "cross-exchange": "cross exchange", "cross-check": "cross check",
    "adaptive-markets": "adaptive markets", "natural-looking": "natural looking",
    "risk-managed": "risk managed", "broad-based": "broad based", "fast-changing": "fast changing",
    "cost-sensitive": "cost sensitive", "single-digit": "single digit", "from-scratch": "from scratch",
    "pair-windows": "pair windows", "random-pair": "random pair", "four-symbol": "four symbol",
    "impact-aware": "impact aware", "within-second": "subsecond", "intra-collapse": "intracollapse",
    "five-minute": "five minute", "round-trip": "round trip", "look-ahead": "look ahead",
    "post-hoc": "post hoc", "non-zero": "nonzero", "state-space": "state space",
    "Month-long": "Month long", "month-long": "month long", "Order-flow": "Order flow",
    "Book-side": "Book side", "Signal-timed": "Signal timed", "signal-timed": "signal timed",
    "Static-OLS": "Static OLS", "Level-2": "Level 2", "per-pair": "per pair", "per-leg": "per leg",
    "per-order": "per order", "per-window": "per window", "per-event": "per event",
    "one-step": "one step", "one-hour": "one hour", "co-moves": "comoves", "at-best": "at best",
    "time-varying": "time varying",
}
for k in sorted(PAIRS, key=len, reverse=True):
    body = re.sub(r"(?<![A-Za-z])" + re.escape(k) + r"(?![A-Za-z])", PAIRS[k], body)

# any remaining numeric en-dash ranges -> "to"
body = re.sub(r"(\$?[+-]?\d[\d.,]*\\?%?\$?)--(\$?[+-]?\d[\d.,]*\\?%?\$?)", r"\1 to \2", body)

open(P, "w", encoding="utf-8", newline="\n").write(body + bib)

print("=== remaining in body (should be only the two label keys) ===")
for h in sorted(set(re.findall(r"[A-Za-z]{2,}-[A-Za-z]{2,}", body))):
    print("  hyphen:", h)
for h in sorted(set(re.findall(r"--", body))):
    print("  en-dash:", repr(h))
# show context of any leftover en-dashes
for m in re.finditer(r".{25}--.{25}", body):
    print("   ctx:", m.group(0).replace("\n", " "))

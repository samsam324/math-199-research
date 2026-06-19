"""Third pass: reproducibility-audit fixes + advisor self-containedness.

- T1 venue table: the "lag 240" HAC label is impossible for a 58-point monthly series (the
  HAC was lag 3-6 on monthly returns); relabel "monthly" and show the Binance HAC as the
  lag-3/6 midpoint ~1.4 (re-runs as 1.49/1.30 via independent_verify.py).
- Block-bootstrap appendix: describe the CIRCULAR variant actually implemented in code.
- Placebo appendix: the Kalman screen's nulls are random-walk / phase-randomized / block-shuffled
  (random-pair is the rolling-z null), fix the attribution.
- Advisor "self-contained for non-experts": add appendix cross-references at first use of each
  tool, a global pointer to Appendix A, and inline definitions of Spearman rho and the Sharpe ratio.
Run: python scratch/review_fixes3.py
"""
P = r"C:\Users\jackw\Desktop\math-199-research\paper\main.tex"
src = open(P, encoding="utf-8").read()


def sub(old, new, label):
    global src
    c = src.count(old)
    assert c == 1, f"[{label}] expected 1, found {c}"
    src = src.replace(old, new)
    print(f"  ok: {label}")


# --- T1 venue table: fix impossible lag label and false-precision HAC value ---
sub(r"Series & Naive monthly Sharpe & HAC-adjusted (lag $240$) \\",
    r"Series & Naive monthly Sharpe & HAC-adjusted (monthly) \\", "T1 lag label")
sub(r"Binance.US, hourly returns & $2.18$ & $1.40$ \\",
    r"Binance.US, hourly returns & $2.18$ & $\approx 1.4$ \\", "T1 Binance HAC value")

# --- block bootstrap appendix: circular variant (matches code) ---
sub(r"$\{1, \dots, T - L + 1\}$, concatenate them, and truncate to length $T$.",
    r"$\{1, \dots, T\}$ with indices taken modulo $T$ (the circular block bootstrap we use), "
    r"concatenate them, and truncate to length $T$.", "block bootstrap circular")

# --- placebo appendix: correct the Kalman-screen null attribution ---
sub(r"fails the random-walk null and the random-pair null at the same rate as the real data,",
    r"fails the random-walk, phase-randomized, and block-shuffled nulls at the same rate as the real data,",
    "placebo attribution")

# --- advisor: appendix cross-references at first use + a global pointer ---
sub(r"errors~\cite{newey1987hac}, because",
    r"errors~\cite{newey1987hac} (Appendix~\ref{app:hac}), because", "xref HAC")
sub(r"six month train and three month test, with",
    r"six month train and three month test (Appendix~\ref{app:wf}), with", "xref walk-forward")
sub(r"mechanical. Selection across many configurations is corrected with a deflated Sharpe",
    r"mechanical (Appendix~\ref{app:placebo}). Selection across many configurations is corrected with a deflated Sharpe",
    "xref placebo")
sub(r"ratio~\cite{bailey2014dsr}.",
    r"ratio~\cite{bailey2014dsr} (Appendix~\ref{app:dsr}), which discounts the Sharpe ratio (a "
    r"strategy's mean return divided by its return standard deviation) for the number of "
    r"configurations searched. Appendix~\ref{sec:appendix} gives self-contained definitions of "
    r"every method and metric used in this paper, for readers who have not met all of them.",
    "xref DSR + Sharpe def + global pointer")
sub(r"A Kalman dynamic hedge lets",
    r"A Kalman dynamic hedge (Appendix~\ref{app:kalman}) lets", "xref Kalman")
sub(r"them~\cite{saiddickey1984}.",
    r"them~\cite{saiddickey1984} (Appendix~\ref{app:adf}).", "xref ADF")
sub(r"a pair's in sample OU reversion speed",
    r"a pair's in sample OU reversion speed (Appendix~\ref{app:ou})", "xref OU")
sub(r"Spearman $\rho = 0.46$ (95\% CI $[0.37, 0.54]$",
    r"Spearman rank correlation $\rho = 0.46$ (95\% CI $[0.37, 0.54]$", "Spearman definition")

open(P, "w", encoding="utf-8", newline="\n").write(src)
print("wrote", P)

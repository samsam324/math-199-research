"""Second factual-correctness pass on paper/main.tex (post full audit).

- tab:exec and order count reconciled to the all-2024 run (29,952 orders), so the headline
  execution table agrees with the +0.03 correlation, the per-symbol table, and the
  "all twelve months" claim (previously it was a January-only 18,432-order subsample).
- oracle saving 1.6 -> 1.4 bps (the all-2024 swing).
- no-breaker zero-crossing 46% -> 50%/yr (survivorship_adjusted_sharpe.csv implies ~50).
- institutional reversal: drop the uncommitted "0.03%" figure, state it qualitatively.
- prereg leg upper bound 520 -> 517 (internal consistency).
- Fig 6 caption extended to the 300s horizons.
- bibliography trimmed 25 -> 23: drop do2010simple (peripheral) and phillips1988testing
  (the appendix named a Phillips-Perron robustness check whose results the paper never reports).
Run: python scratch/review_fixes2.py
"""
P = r"C:\Users\jackw\Desktop\math-199-research\paper\main.tex"
src = open(P, encoding="utf-8").read()


def sub(old, new, label, n=1):
    global src
    c = src.count(old)
    assert c == n, f"[{label}] expected {n}, found {c}"
    src = src.replace(old, new)
    print(f"  ok: {label}")


# --- tab:exec -> all-2024 (29,952-order) pooled costs ---
sub(r"Aggressive (always cross) & $+1.34$ \\", r"Aggressive (always cross) & $+1.10$ \\", "exec agg")
sub(r"Naive passive (always post) & $+2.79$ \\", r"Naive passive (always post) & $+2.69$ \\", "exec naive")
sub(r"Signal timed (post/cross on the L2 signal) & $+1.98$ \\",
    r"Signal timed (post/cross on the L2 signal) & $+1.73$ \\", "exec signal")
sub(r"Random at the same posting rate & $+1.88$ \\", r"Random at the same posting rate & $+1.68$ \\", "exec random")
sub(r"Oracle (perfect foresight) & $-0.23$ \\", r"Oracle (perfect foresight) & $-0.30$ \\", "exec oracle")

# --- order count 18,432 (Jan-only) -> 29,952 (all-2024), all 4 occurrences ---
sub(r"18{,}432", r"29{,}952", "order count 18432->29952", n=4)

# --- oracle saving 1.6 -> 1.4 bps (two phrasings) ---
sub(r"oracle would save $1.6$ bps", r"oracle would save $1.4$ bps", "oracle bps intro")
sub(r"oracle would save 1.6 bps,", r"oracle would save 1.4 bps,", "oracle bps body")

# --- no-breaker zero crossing 46 -> 50 %/yr ---
sub(r"crosses zero only near a $46\%$/year", r"crosses zero only near a $50\%$/year", "breaker crossing 46->50")

# --- institutional reversal: drop uncommitted 0.03% ---
sub(r"continuation, with an incremental $R^2$ near $0.03\%$.",
    r"continuation, with a negligible incremental $R^2$.", "inst R2 qualitative")

# --- prereg leg upper bound 520 -> 517 (consistency with the +517% leg cited next sentence) ---
sub(r"post $+300$ to $+520\%$ in three months", r"post $+300$ to $+517\%$ in three months", "prereg leg 520->517")

# --- Fig 6 caption: extended horizons ---
sub(r"to near zero by $30$ seconds, across BTC/ETH/SOL/AVAX",
    r"to under $0.3\%$ at one second and a $0.01$ to $0.06\%$ noise floor out to $300$ seconds, "
    r"across BTC/ETH/SOL/AVAX", "ofi caption extended")

# --- bibliography trim to 23 ---
# drop do2010simple (entry + in-text cite)
sub("\n".join([r"\bibitem{do2010simple}",
               r"Do, B., \& Faff, R. (2010). Does simple pairs trading still work?",
               r"\emph{Financial Analysts Journal}, 66(4), 83--95.", "", ""]),
    "", "drop do2010 bibitem")
sub(r"benchmark; Do and Faff~\cite{do2010simple} document its secular decay on US equities.",
    r"benchmark.", "drop do2010 cite")
# drop phillips1988testing (entry + the unreported PP robustness sentence)
sub("\n".join([r"\bibitem{phillips1988testing}",
               r"Phillips, P.~C.~B., \& Perron, P. (1988). Testing for a unit root in time series",
               r"regression. \emph{Biometrika}, 75(2), 335--346.", "", ""]),
    "", "drop phillips bibitem")
sub("\n".join([r"$t_{\hat\rho}$. We supplement ADF with the nonparametric Phillips-Perron",
               r"test~\cite{phillips1988testing} for robustness."]),
    r"$t_{\hat\rho}$.", "drop phillips cite+sentence")

open(P, "w", encoding="utf-8", newline="\n").write(src)
print("wrote", P)

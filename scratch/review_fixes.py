"""Apply the adversarial-review fixes to paper/main_v2.tex.

Factual: +0.06 exec correlation -> +0.03 (committed log pooled value); OFI contemporaneous
range +0.12-0.16 -> +0.11-0.15 (ETH is 0.1137); microcap floor "hundreds" -> "tens" (WAXP $92).
Caption: circuit-breaker "collapses past 10%/yr" -> crosses zero near 46%/yr; HAC "1.5-2.5x" ->
"around 2x"; cancel "about 80%" -> "77 to 80%"; appendix cross-ref noun fix.
Voice: drop redundant maxim restatement; de-cliche the Related Work opener.
Run: python scratch/review_fixes.py
"""
P = r"C:\Users\jackw\Desktop\math-199-research\paper\main_v2.tex"
src = open(P, encoding="utf-8").read()


def sub(old, new, label):
    global src
    c = src.count(old)
    assert c == 1, f"[{label}] expected 1, found {c}"
    src = src.replace(old, new)
    print(f"  ok: {label}")


# --- factual ---
sub(r"the signal correlates $+0.06$ with the per-order post-vs-cross",
    r"the signal correlates $+0.03$ with the per-order post-vs-cross", "abstract corr +0.06->+0.03")
sub(r"correlation with the per order advantage of posting over crossing is $+0.06$, noise.",
    r"correlation with the per order advantage of posting over crossing is $+0.03$, noise.", "body corr +0.06->+0.03")
sub(r"better than trade flow does (incremental $R^2$",
    r"better than trade flow does (contemporaneous incremental $R^2$", "ofi body unit")
sub(r"of $+0.12$ to $+0.16$ for BTC and ETH)",
    r"of $+0.11$ to $+0.15$ for BTC and ETH)", "ofi body range")
sub(r"trade at hundreds to a few thousand dollars per day, three to",
    r"trade at tens to a few thousand dollars per day, three to", "microcap caption floor")
sub(r"turn over a median of a few hundred to a few",
    r"turn over a median of tens to a few", "microcap body floor")

# --- caption corrections ---
sub("\n".join([r"Without the breaker, the Sharpe collapses", r"past a $10\%$/year break rate."]),
    "\n".join([r"Without the breaker, the Sharpe declines steadily and crosses zero only near a $46\%$/year",
               r"break rate."]), "circuit-breaker caption")
sub(r"cluster around $1.5$--$2.5\times$ across the $27$",
    r"cluster around $2\times$ (median $2.0$) across the $27$", "hac caption cluster")
sub(r"(about $80\%$ for BTC, ETH, SOL; $90\%$ for AVAX)",
    r"(about $77$ to $80\%$ for BTC, ETH, SOL; $90\%$ for AVAX)", "cancel caption range")
sub(r"The connection to the rolling-$z$ artifact (Section~\ref{sec:kalman})",
    r"The connection to the Kalman innovation artifact (Section~\ref{sec:kalman})", "appendix xref noun")

# --- voice ---
sub("\n".join([r"The lesson is narrow and reusable: before you believe a cointegration or mean reversion screen,",
               r"check that it fails on a matched placebo. The backtest overfitting"]),
    r"The backtest overfitting", "cut redundant maxim")
sub("\n".join([
    r"Our work sits at the intersection of three literatures: the cointegration branch of pairs",
    r"trading, modern market microstructure, and empirical studies of cryptocurrency market",
    r"efficiency. The pairs literature provides the Kalman dynamic hedge and rolling $z$-score",
    r"screens whose artifacts we expose; the microstructure literature provides the order-flow",
    r"machinery we apply to crypto L2 data; and the crypto-efficiency literature is the context",
    r"in which our headline negative result lands.",
]), "\n".join([
    r"Three bodies of prior work set up this paper. The two screens we audit come from the",
    r"cointegration branch of the pairs-trading literature. The order-flow tools we apply to crypto",
    r"L2 data are standard market microstructure. And the crypto market-efficiency literature is",
    r"where our headline negative result has to be read.",
]), "related-work opener")

open(P, "w", encoding="utf-8", newline="\n").write(src)
print("wrote", P)

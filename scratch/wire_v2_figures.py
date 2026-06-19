"""Wire the generated figures and table values into paper/main_v2.tex.

Deterministic: each replacement asserts it fired so a stale string fails loudly.
Run: python scratch/wire_v2_figures.py
"""
import re

P = r"C:\Users\jackw\Desktop\math-199-research\paper\main_v2.tex"
src = open(P, encoding="utf-8").read()


def sub(old, new, n=1, label=""):
    global src
    c = src.count(old)
    assert c == n, f"expected {n} of [{label or old[:40]}], found {c}"
    src = src.replace(old, new)


# --- figure placeholders -> includegraphics ---
FIGS = [
    (r"\figplaceholder{fig\_kalman\_innovations\_white.pdf}{scratch/kalman\_positive\_control.py (re-run to dump innovations)}",
     r"\includegraphics[width=\linewidth]{fig_kalman_innovations_white}"),
    (r"\figplaceholder{fig\_poscontrol\_vs\_negcontrol.pdf}{scratch/kalman\_positive\_control.csv}",
     r"\includegraphics[width=0.82\linewidth]{fig_poscontrol_vs_negcontrol}"),
    (r"\figplaceholder{fig\_freq\_invariance.pdf (RESHAPED)}{table values reshaped to show placebo gap}",
     r"\includegraphics[width=0.74\linewidth]{fig_freq_invariance}"),
    (r"\figplaceholder{fig\_reversion\_persistence\_scatter.pdf}{scratch/persistence\_pairs.csv}",
     r"\includegraphics[width=\linewidth]{fig_reversion_persistence_scatter}"),
    (r"\figplaceholder{fig\_ofi\_decay.pdf}{re-run scratch/book\_ofi\_incremental.py with horizons [1,2,5,10,30,60,120,300] s --- requires L2 data on Windows}",
     r"\includegraphics[width=\linewidth]{fig_ofi_decay}"),
    (r"\figplaceholder{fig\_cancellation\_share.pdf}{scratch/book\_ofi\_2024.log cancel-share fields --- requires L2 data on Windows}",
     r"\includegraphics[width=0.66\linewidth]{fig_cancellation_share}"),
    (r"\figplaceholder{fig\_hac\_inflation.pdf}{docs/hac\_sharpe\_per\_split.csv}",
     r"\includegraphics[width=\linewidth]{fig_hac_inflation}"),
    (r"\figplaceholder{fig\_microcap\_adv.pdf}{scratch/microcap\_adv.csv}",
     r"\includegraphics[width=0.82\linewidth]{fig_microcap_adv}"),
    (r"\figplaceholder{fig\_forced\_collapse\_perpair.pdf}{scratch/forced\_collapse.csv}",
     r"\includegraphics[width=\linewidth]{fig_forced_collapse_perpair}"),
    (r"\figplaceholder{fig\_circuit\_breaker\_retention.pdf}{scratch/survivorship\_adjusted\_sharpe.csv}",
     r"\includegraphics[width=0.8\linewidth]{fig_circuit_breaker_retention}"),
]
for old, new in FIGS:
    sub(old, new, label=old[:45])

# --- T2 cells (exec by symbol): Agg/Signal-timed at $10k and $50k, l3_30 horizon ---
sub(r"BTCUSDT & TBD & TBD & TBD & TBD \\",  r"BTCUSDT & $0.02$ & $0.64$ & $0.09$ & $0.71$ \\", label="T2 BTC")
sub(r"ETHUSDT & TBD & TBD & TBD & TBD \\",  r"ETHUSDT & $0.08$ & $0.80$ & $0.26$ & $0.94$ \\", label="T2 ETH")
sub(r"SOLUSDT & TBD & TBD & TBD & TBD \\",  r"SOLUSDT & $0.56$ & $1.36$ & $1.28$ & $2.01$ \\", label="T2 SOL")
sub(r"AVAXUSDT & TBD & TBD & TBD & TBD \\", r"AVAXUSDT & $2.09$ & $2.70$ & $4.45$ & $4.69$ \\", label="T2 AVAX")

# --- strip the three "[PLACEHOLDER ... ]" caption prefixes ---
n_before = len(re.findall(r"\[PLACEHOLDER", src))
assert n_before == 3, f"expected 3 placeholder caption prefixes, found {n_before}"
src = re.sub(r"\[PLACEHOLDER --- to be populated from [^\]]*\]\s*", "", src)
assert "[PLACEHOLDER" not in src

# --- T1 venue table: replace the em-dash cell with n/a ---
sub(r"& --- & $\approx 1.0$", r"& n/a & $\approx 1.0$", label="T1 emdash cell")

# --- caption corrections where written range != data (single-line tokens) ---
# Fig 6 ofi-decay: contemporaneous incr R^2 runs to 48% (AVAX), not 10-15%.
sub(r"$\sim$$10$--$15\%$ contemporaneously",
    r"$\sim$$11$ to $48\%$ contemporaneously", label="ofi caption range")

# Fig 9 cancel-share: data is 77-90%, not 81-85%.
sub(r"withdrawal: $81$--$85\%$ of",
    r"withdrawal: $77$ to $90\%$ (about $80\%$ for BTC, ETH, SOL; $90\%$ for AVAX) of",
    label="cancel caption range")

# Reshaped freq figure: disclose the clean-test placebo baseline.
sub(r"frequency dependent. This reshaped view",
    r"frequency dependent, taking the clean test's random walk placebo at its nominal $5\%$ size. "
    r"This reshaped view", label="freq caption baseline")

# T2 caption: claim only what the table shows (no random-rate column here).
sub(r"; the L3-from-L2 signal never beats the",
    r": the L3-from-L2 signal-timed rule is more expensive than crossing in every", label="T2 caption a")
sub(r"random-rate placebo at any cell.",
    r"cell, consistent with the pooled result of Table~\ref{tab:exec}.", label="T2 caption b")

# Body: cancellation range to match the figure.
sub(r"$81$ to $85\%$", r"$77$ to $90\%$", label="body cancel range")

open(P, "w", encoding="utf-8", newline="\n").write(src)

# report
left = re.findall(r"figplaceholder\{", src)
print("remaining figplaceholder uses:", len(left))
print("remaining TBD cells:", src.count("& TBD &"))
print("remaining --- (em-dash) in file:", len(re.findall(r"(?<!\d)---(?!\d)", src)))
print("includegraphics count:", src.count(r"\includegraphics"))
print("wrote", P)

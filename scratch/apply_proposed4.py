"""Fourth pass on main_proposed.tex: referee polish on the L2-breadth additions only
(in scope). Pre-existing canonical-paper issues (survivor Sharpe CI, venue ~1.0,
single-t0 Kalman) are NOT touched here; they are surfaced to the authors separately.
Run AFTER apply_proposed3.py:  python scratch/apply_proposed4.py
"""
P = r"C:\Users\jackw\Desktop\math-199-research\main_proposed.tex"
src = open(P, encoding="utf-8").read()


def sub(old, new, label):
    global src
    c = src.count(old)
    assert c == 1, f"[{label}] expected 1 match, found {c}"
    src = src.replace(old, new)
    print("  ok:", label)


# A. make explicit the breadth R^2 is the JOINT (trade+book) forecast R^2, the generous bound
sub("yet the one second ahead predictive $R^2$ stays under $0.5\\%$ for every one of the $50$ symbols",
    "yet the one second ahead forecast $R^2$, taking trade and book OFI together, stays under "
    "$0.5\\%$ for every one of the $50$ symbols",
    "A: clarify joint forecast R2")

# B. name the two negative-estimate symbols
sub("and two\nsymbols show a negative estimate.",
    "and two\nsymbols (NMR and SAND) show a negative institutional estimate.",
    "B: name negative-impact symbols")

# C. scope Table 6 caption to the four megacaps + point to the 50-symbol breakdown
sub("institutional\n($>$\\$$10$k) consistently $2.0$ to $2.6\\times$ retail across BTC/ETH/SOL/AVAX, validating\n"
    "size as a per order information proxy.",
    "institutional\n($>$\\$$10$k) runs $2.0$ to $2.6\\times$ retail across these four megacaps, where size\n"
    "proxies information; across the full top 50 the ratio weakens to a median $1.5\\times$ and turns noisy\n"
    "on thin names (Section~\\ref{sec:real}).",
    "C: scope Table 6 caption to megacaps")

# D. intro: note the execution null extends to the full top 50
sub("no contemporaneous feature forecasts it.",
    "no contemporaneous feature forecasts it, and the same null holds when the simulation is rerun "
    "across all fifty symbols.",
    "D: intro execution-breadth clause")

open(P, "w", encoding="utf-8", newline="\n").write(src)
print("wrote", P)

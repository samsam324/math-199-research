"""Fifth pass on main_proposed.tex: fold in the Kalman multi-anchor robustness
(closes the referee's "single start date" concern on the flagship Table 1).
Adds a short paragraph + a 5-anchor table with Wilson CIs after Table 1.
Numbers from scratch/kalman_anchors.csv (book_kalman_anchors.py, 5 quarterly t0).
Run AFTER apply_proposed4.py:  python scratch/apply_proposed5.py
"""
P = r"C:\Users\jackw\Desktop\math-199-research\main_proposed.tex"
src = open(P, encoding="utf-8").read()

NEW = (
    "The artifact does not depend on that single start date. We re-ran the identical screen at five\n"
    "disjoint quarterly anchors from mid 2023 to mid 2024 (Table~\\ref{tab:kalman-anchors}). At every\n"
    "one, real pairs and all three placebos pass the Kalman innovation ADF at $99$ to $100\\%$, with\n"
    "Wilson $95\\%$ intervals that never fall below $95\\%$, while the clean Engle-Granger benchmark\n"
    "stays between $1$ and $8\\%$. The screen reports the same near total cointegration whether the\n"
    "input can be cointegrated or not, at every start date we tried.\n\n"
    "\\begin{table}[h]\n\\centering\n"
    "\\caption{The artifact is not specific to one start date: the identical screen at five disjoint\n"
    "quarterly anchors. Real pairs and all three placebos (independent random walk, phase randomized,\n"
    "block shuffled) pass the Kalman innovation ADF at $p<0.05$ at nearly $100\\%$ (Wilson $95\\%$\n"
    "intervals in brackets; $n=120$ real, $80$ per placebo), while the clean Engle-Granger test stays\n"
    "low. $90$ day train / $30$ day test at each $t_0$.}\n"
    "\\label{tab:kalman-anchors}\n"
    "\\begin{tabular}{lrrr}\n\\toprule\n"
    "Anchor $t_0$ & Real Kalman & All placebos & Clean EG \\\\\n\\midrule\n"
    "2023-07-01 & $100\\%$ $[97,100]$ & $100\\%$ & $3.8\\%$ \\\\\n"
    "2023-10-01 & $99.2\\%$ $[95,100]$ & $100\\%$ & $1.4\\%$ \\\\\n"
    "2024-01-01 & $100\\%$ $[97,100]$ & $100\\%$ & $2.1\\%$ \\\\\n"
    "2024-04-01 & $100\\%$ $[97,100]$ & $100\\%$ & $8.1\\%$ \\\\\n"
    "2024-07-01 & $100\\%$ $[97,100]$ & $100\\%$ & $6.4\\%$ \\\\\n"
    "\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")

old = "We also ran a positive control:"
assert src.count(old) == 1, f"anchor expected 1, found {src.count(old)}"
src = src.replace(old, NEW + old)
open(P, "w", encoding="utf-8", newline="\n").write(src)
print("ok: inserted Kalman multi-anchor robustness paragraph + table")

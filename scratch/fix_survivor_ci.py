"""Correctness fix applied to BOTH main.tex and main_proposed.tex: the pre-registered
single-window survivor reported "a monthly Sharpe of 1.60 with a HAC confidence
interval of [-0.21, 1.89]". Per the committed scratch/prereg_run.log, that CI belongs
to the HAC hourly-annualized Sharpe of 0.95, NOT the monthly Sharpe of 1.60 (two
different conventions). Re-attribute it. The conclusion (CI spans zero, no significant
edge) is unchanged. Run: python scratch/fix_survivor_ci.py
"""
OLD = ("a monthly Sharpe of $1.60$ with a HAC confidence interval of $[-0.21,\n"
       "1.89]$ that spans zero")
NEW = ("a monthly Sharpe of $1.60$, with a HAC hourly annualized Sharpe of $0.95$ whose $95\\%$\n"
       "confidence interval $[-0.21, 1.89]$ spans zero")

for P in (r"C:\Users\jackw\Desktop\math-199-research\main.tex",
          r"C:\Users\jackw\Desktop\math-199-research\main_proposed.tex"):
    src = open(P, encoding="utf-8").read()
    c = src.count(OLD)
    assert c == 1, f"[{P}] expected 1 match, found {c}"
    open(P, "w", encoding="utf-8", newline="\n").write(src.replace(OLD, NEW))
    print("  fixed:", P)

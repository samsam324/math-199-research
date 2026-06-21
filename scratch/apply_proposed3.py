"""Third pass on main_proposed.tex: fix the issues the adversarial verification found.
 A. "2.8% on the majors" was the universe-wide average, not the four majors (~4-6%).
 B. clarify the 1.4 bps oracle is the FOUR majors (matches the paper's existing 1.4).
 C. soften "loses to a random post or cross" to the aggregate (48/50, not literally all).
 D. USDC footnote: it is the only symbol that rises MATERIALLY (others rise trivially).
Run AFTER apply_proposed2.py:  python scratch/apply_proposed3.py
"""
P = r"C:\Users\jackw\Desktop\math-199-research\main_proposed.tex"
src = open(P, encoding="utf-8").read()


def sub(old, new, label):
    global src
    c = src.count(old)
    assert c == 1, f"[{label}] expected 1 match, found {c}"
    src = src.replace(old, new)
    print("  ok:", label)


# A. the 2.8% was the universe-wide institutional-order share, not the majors' (1.8-6.3%)
sub("are only $0.2\\%$ of orders against $2.8\\%$ on the majors and two",
    "are only $0.2\\%$ of orders, roughly an order of magnitude rarer than on the majors, and two",
    "A: fix mislabeled 2.8% majors share")

# B. the 1.4 bps oracle is specifically the four event-level majors
sub("several times the $1.4$ on the majors",
    "several times the $1.4$ on the four majors",
    "B: clarify 1.4 = four majors")

# C. signal-timed rule loses to the random post/cross placebo in aggregate (48/50), not literally all
sub("(median $0.4$ bps worse) and loses to a\nrandom post or cross.",
    "(median $0.4$ bps worse) and on the whole loses to a\nrandom post or cross.",
    "C: soften loses-to-placebo to aggregate")

# D. USDC is the only symbol whose forecast R^2 rises MATERIALLY (a handful rise trivially)
sub("The one exception to the monotone decay is USDC",
    "The only symbol whose forecast $R^2$ rises materially with horizon is USDC",
    "D: USDC rises materially, not lone non-monotone")

open(P, "w", encoding="utf-8", newline="\n").write(src)
print("wrote", P)

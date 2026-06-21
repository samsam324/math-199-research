"""Second pass on main_proposed.tex: add the execution + size-to-impact breadth
results (the OFI/cancellation breadth is already in from apply_proposed.py).
Execution strengthens cleanly across all 50; size-to-impact is honest: liquid-core
only. Each edit asserts a unique match. Run AFTER apply_proposed.py.
    python scratch/apply_proposed2.py
"""
P = r"C:\Users\jackw\Desktop\math-199-research\main_proposed.tex"
src = open(P, encoding="utf-8").read()


def sub(old, new, label):
    global src
    c = src.count(old)
    assert c == 1, f"[{label}] expected 1 match, found {c}"
    src = src.replace(old, new)
    print("  ok:", label)


# ---- 1. new paragraph (execution + size-to-impact breadth) after the OFI breadth paragraph ----
NEW_PARA2 = r"""

The execution and size to impact results extend the same way, with one wrinkle. Rerunning the order
level execution simulation on all fifty symbols leaves the null intact: the signal timed rule is
more expensive than aggressive crossing in every one of them (median $0.4$ bps worse) and loses to a
random post or cross. On the thinnest names a perfect foresight oracle could save $4$ to $5$ bps,
several times the $1.4$ on the majors, yet the contemporaneous signal correlates at most $+0.07$ with
that advantage, so the larger execution opportunity on thin books is just as unforecastable. Size to
impact is the one result that does not fully generalize: institutional one second impact exceeds
retail on all ten of the most liquid symbols and on $39$ of $50$ overall, but the ratio falls to a
median $1.5\times$ from the $2.0$ to $2.6\times$ on the majors and turns noisy on the thinnest names,
where institutional sized trades are only $0.2\%$ of orders against $2.8\%$ on the majors and two
symbols show a negative estimate. Trade size proxies information where there is enough institutional
flow to measure it, which is the liquid core."""
sub("mildly non-monotone at trivial magnitudes below $0.3\\%$.}",
    "mildly non-monotone at trivial magnitudes below $0.3\\%$.}" + NEW_PARA2,
    "execution+impact breadth paragraph")

# ---- 2. limitations bullet: OFI/cancel/execution now top-50; impact liquid-only ----
sub("The book-OFI decay and cancellation results cover the full top 50 universe, over days\n"
    "sampled across 2024 rather than a continuous year. The execution and impact tests remain the\n"
    "four majors (BTC, ETH, SOL, AVAX) only; we have not extended the order level execution\n"
    "simulation or the impact decomposition to the wider universe.",
    "The book-OFI decay, cancellation, and execution results cover the full top 50 universe, over\n"
    "days sampled across 2024 rather than a continuous year. The size to impact ranking holds on the\n"
    "liquid names (all ten of the most liquid, $39$ of $50$ overall) but turns noisy on the thinnest,\n"
    "where institutional sized trades are rare, so the impact table stays the four majors.",
    "limitations bullet v2")

# ---- 3. Section 5 scope sentence ----
sub("The Level 2 tests cover the full top 50 for the order flow and cancellation null and four\n"
    "megacap symbols for execution and impact, all on the global venue.",
    "The Level 2 tests cover the full top 50 for the order flow, cancellation, and execution nulls,\n"
    "and the four megacap symbols for the size to impact table, all on the global venue.",
    "section 5 scope sentence v2")

open(P, "w", encoding="utf-8", newline="\n").write(src)
print("wrote", P)

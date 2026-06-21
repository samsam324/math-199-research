"""Build main_proposed.tex from main.tex with the 50-symbol L2 breadth additions.
Leaves main.tex untouched. Each edit asserts a unique match. Run:
    python scratch/apply_proposed.py
"""
SRC = r"C:\Users\jackw\Desktop\math-199-research\main.tex"
DST = r"C:\Users\jackw\Desktop\math-199-research\main_proposed.tex"
src = open(SRC, encoding="utf-8").read()


def sub(old, new, label):
    global src
    c = src.count(old)
    assert c == 1, f"[{label}] expected 1 match, found {c}"
    src = src.replace(old, new)
    print("  ok:", label)


# ---- A. abstract: broaden the order-flow decay sentence ----
sub("Level 2 book decays within seconds.",
    "Level 2 book decays within seconds, and that null holds across all fifty of the most liquid symbols.",
    "abstract decay breadth")

# ---- B. Section 5: new universe-wide paragraph after the 4-symbol discussion ----
NEW_PARA = r"""

We then extended the book-OFI test from the four event level majors to the full top 50
universe over the same regime spanning days, and the null is the same across all of them. Book-OFI
adds $3$ to $48\%$ to the contemporaneous one second $R^2$ over trade-OFI alone (median $23\%$,
lowest on the thinnest books, where trade flow already captures what little information is there),
yet the one second ahead predictive $R^2$ stays under $0.5\%$ for every one of the $50$ symbols
(highest $0.46\%$ for BTC, median $0.13\%$) and falls toward zero by thirty seconds
(Figure~\ref{fig:ofi-breadth}). The thinnest names, where slower price formation could have left a
tradable window, carry the least forecast power of all. Cancellations run $62$ to $97\%$ of best
quote liquidity withdrawal (median $90\%$).\footnote{The one exception to the monotone decay is USDC
against USDT, the lone stablecoin in the set, whose forecast $R^2$ rises to $0.6\%$ at thirty
seconds; this reflects peg mean reversion rather than a directional signal. A handful of thin names
are mildly non-monotone at trivial magnitudes below $0.3\%$.}"""
sub("invisible to trade only analysis (Figure~\\ref{fig:cancel-share}).",
    "invisible to trade only analysis (Figure~\\ref{fig:cancel-share})." + NEW_PARA,
    "section 5 breadth paragraph")

# ---- C. new breadth figure, placed after the existing 4-symbol decay figure ----
NEW_FIG = r"""

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{fig_ofi_breadth}
\caption{Order flow across the full top 50 universe, one point per symbol, over the same regime
spanning 2024 days. Horizontal axis: how much book-OFI adds to the contemporaneous one second $R^2$
over trade-OFI alone ($3$ to $48\%$). Vertical axis: the one second ahead forecast $R^2$. Left, on a
shared scale, the forecast power sits on the floor; right, with the forecast axis magnified about
$100\times$, it stays under $0.5\%$ for every symbol and bears no relation to contemporaneous
strength. Order flow is real but priced almost immediately, across the whole liquid core, not only
the four event level majors (red).}
\label{fig:ofi-breadth}
\end{figure}"""
sub("\\label{fig:ofi-decay}\n\\end{figure}",
    "\\label{fig:ofi-decay}\n\\end{figure}" + NEW_FIG,
    "breadth figure float")

# ---- D. Figure 6 caption: note the decay generalizes ----
sub("all 2024 market regimes. The information half life is seconds",
    "all 2024 market regimes. The same decay holds across the full top 50 universe "
    "(Section~\\ref{sec:real}). The information half life is seconds",
    "fig6 caption breadth clause")

# ---- E. limitations bullet: coverage now top-50 for OFI/cancel, 4 for exec/impact ----
sub("The event level Level 2 tests use four symbols (BTC, ETH,\n"
    "SOL, AVAX) over days sampled across 2024, not a continuous year on the full universe. The\n"
    "aggregate book metrics use the top 50. The execution and impact results are four symbol.",
    "The book-OFI decay and cancellation results cover the full top 50 universe, over days\n"
    "sampled across 2024 rather than a continuous year. The execution and impact tests remain the\n"
    "four majors (BTC, ETH, SOL, AVAX) only; we have not extended the order level execution\n"
    "simulation or the impact decomposition to the wider universe.",
    "limitations microstructure bullet")

# ---- F. Section 5 scope sentence: reflect the broadened coverage ----
sub("The Level 2 tests cover top 50 aggregate book\nmetrics and four megacap symbols on the global venue.",
    "The Level 2 tests cover the full top 50 for the order flow and cancellation null and four\n"
    "megacap symbols for execution and impact, all on the global venue.",
    "section 5 scope sentence")

open(DST, "w", encoding="utf-8", newline="\n").write(src)
print("wrote", DST)

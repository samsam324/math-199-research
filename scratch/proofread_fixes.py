"""Apply the copy-editor's must-fix items to paper/main.tex (garbled sentences, garden-path
de-hyphen artifacts rephrased, British spelling, one duplicated clause). Run after dedash_v2.py.
Run: python scratch/proofread_fixes.py"""
P = r"C:\Users\jackw\Desktop\math-199-research\paper\main.tex"
src = open(P, encoding="utf-8").read()


def sub(old, new, label):
    global src
    c = src.count(old)
    assert c == 1, f"[{label}] expected 1, found {c}"
    src = src.replace(old, new)
    print(f"  ok: {label}")


# 1. parenthetical comma splice -> semicolon
sub(r"(Table~\ref{tab:rollingz}, Figure~\ref{fig:rollingz} shows the mechanism on one path).",
    r"(Table~\ref{tab:rollingz}; Figure~\ref{fig:rollingz} shows the mechanism on one path).", "rollingz paren splice")

# 2. dangling -ing clause in the freq caption
sub(r"frequency dependent, taking the clean test's random walk placebo at its nominal $5\%$ size. This reshaped view",
    r"frequency dependent. The clean test's random walk placebo passes at its nominal $5\%$ size. This reshaped view",
    "freq caption dangle")

# 3. missing "versus" in the frequency sentence
sub(r"$100\%$ at hourly and four hour, and $70\%$ versus $68\%$ at daily",
    r"$100\%$ versus $100\%$ at hourly and four hour, and $70\%$ versus $68\%$ at daily", "freq missing versus")

# 4. comma splice in the impact sentence
sub(r"$2.6\times$ retail, the longer horizon permanent estimate is noisier;",
    r"$2.6\times$ retail, though the longer horizon permanent estimate is noisier;", "impact comma splice")

# 5. triple-"to" garble in the OFI caption
sub(r"$\sim$$11$ to $48\%$ contemporaneously to under $0.3\%$ at one second",
    r"$\sim$$11$ to $48\%$ contemporaneously down to under $0.3\%$ at a one second forecast horizon",
    "ofi caption triple-to")

# 9. garden-path "risk off waves" -> rephrase
sub(r"which understates clustered risk off waves.",
    r"which understates the way breaks cluster in market wide selloffs.", "risk-off rephrase")

# 10. garden-path "broad based inflation" -> rephrase
sub(r"It is broad based inflation across", r"It is inflation spread broadly across", "broad-based rephrase")

# 19. British -> American spelling (appendix)
sub(r"whitening behaviour for any input", r"whitening behavior for any input", "behaviour->behavior")
sub(r"the filter has been optimised", r"the filter has been optimized", "optimised->optimized")

# 31. garden-path "fast changing set"
sub(r"a fast changing set of listed coins", r"a rapidly changing set of listed coins", "fast-changing rephrase")

# 21. drop undefined "post-vs-cross" jargon (matches the longhand used later in the section)
sub("\n".join([r"with the per order post-vs-cross", r"advantage,"]),
    r"with the per order advantage of posting over crossing,", "post-vs-cross longhand")

# 29. remove the duplicated "limits matter more than the headline" clause (kept once below)
sub(r"One strategy survives, but only barely, and its limits matter more than its headline. There is a real, market neutral,",
    r"One strategy survives, but only barely. There is a real, market neutral,", "dedupe headline clause")

open(P, "w", encoding="utf-8", newline="\n").write(src)
print("wrote", P)

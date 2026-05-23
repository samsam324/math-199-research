# Paper

LaTeX source for the Phase 1 paper.

## Build

```
cd paper
pdflatex paper.tex
pdflatex paper.tex   # second pass for refs
```

If you want figures inline, copy or symlink the relevant figures from
the project's `figures/` directory into `paper/figures/`. The paper
currently references:

- `figures/kalman_screen_cdf.png` (the universe-wide OOS CDF)

If `paper/figures/` is missing, the build will still compile but the
figure will be replaced by a placeholder error message.

## Quick setup

```
cd paper
mkdir -p figures
cp ../figures/kalman_screen_cdf.png figures/
pdflatex paper.tex
pdflatex paper.tex
open paper.pdf
```

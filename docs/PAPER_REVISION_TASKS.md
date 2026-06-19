# Paper Revision Tasks (v2)

This is a trimmed and honesty-tightened version of the previous revision plan.
**v1's 60-citation bibliography was overkill** for a three-student undergrad project, and
**v1 oversold the novelty** of mechanisms (Kalman innovation whitening, rolling-mean
demeaning) that are textbook. v2 cuts the bibliography to ~22 entries focused on works
we actually engage with by content, and rewrites the contributions paragraph to be
honest about what is genuinely new (the placebo audits and the L2 execution-value
measurement) vs what is replication or warning.

The methods appendix and figure specifications carry over from v1 with small patches
to drop citations to works that are no longer in the trimmed bibliography.

## Section 0: Summary

Apply in the order of Section 7. Bibliography must merge first because every other
section cites by `\cite{key}`. All cite keys in this document are pre-resolved against
Section 1's bibliography; no consistency notes needed.

---

## Section 1: Bibliography (22 entries total)

### Keep these 4 already in `paper/main.tex`

- `bailey2014dsr`
- `engle1987cointegration`
- `fil2020pairs`
- `gatev2006pairs`

### Add these 19 (sorted by key, ready to drop into `\begin{thebibliography}{99}`)

```latex
\bibitem{andersen2014vpin}
Andersen, T.~G., \& Bondarenko, O. (2014). VPIN and the flash crash.
\emph{Journal of Financial Markets}, 17, 1--46.

\bibitem{brauneis2018price}
Brauneis, A., \& Mestel, R. (2018). Price discovery of cryptocurrencies:
Bitcoin and beyond. \emph{Economics Letters}, 165, 58--61.

\bibitem{contkukanovstoikov2014}
Cont, R., Kukanov, A., \& Stoikov, S. (2014). The price impact of order book events.
\emph{Journal of Financial Econometrics}, 12(1), 47--88.

\bibitem{do2010simple}
Do, B., \& Faff, R. (2010). Does simple pairs trading still work?
\emph{Financial Analysts Journal}, 66(4), 83--95.

\bibitem{easley2012vpin}
Easley, D., L\'opez de Prado, M.~M., \& O'Hara, M. (2012). Flow toxicity and liquidity
in a high-frequency world. \emph{Review of Financial Studies}, 25(5), 1457--1493.

\bibitem{elliott2005pairs}
Elliott, R.~J., van der Hoek, J., \& Malcolm, W.~P. (2005). Pairs trading.
\emph{Quantitative Finance}, 5(3), 271--276.

\bibitem{harvey1989structural}
Harvey, A.~C. (1989). \emph{Forecasting, Structural Time Series Models and the
Kalman Filter}. Cambridge University Press.

\bibitem{hasbrouck1991measuring}
Hasbrouck, J. (1991). Measuring the information content of stock trades.
\emph{Journal of Finance}, 46(1), 179--207.

\bibitem{krauss2017deep}
Krauss, C., Do, X.~A., \& Huck, N. (2017). Deep neural networks, gradient-boosted trees,
random forests: statistical arbitrage on the S\&P~500. \emph{European Journal of
Operational Research}, 259(2), 689--702.

\bibitem{kunsch1989block}
K\"unsch, H.~R. (1989). The jackknife and the bootstrap for general stationary
observations. \emph{Annals of Statistics}, 17(3), 1217--1241.

\bibitem{kyle1985continuous}
Kyle, A.~S. (1985). Continuous auctions and insider trading.
\emph{Econometrica}, 53(6), 1315--1335.

\bibitem{lopezdeprado2018advances}
L\'opez de Prado, M. (2018). \emph{Advances in Financial Machine Learning}. Wiley.

\bibitem{makarov2020trading}
Makarov, I., \& Schoar, A. (2020). Trading and arbitrage in cryptocurrency markets.
\emph{Journal of Financial Economics}, 135(2), 293--319.

\bibitem{newey1987hac}
Newey, W.~K., \& West, K.~D. (1987). A simple, positive semi-definite,
heteroskedasticity and autocorrelation consistent covariance matrix.
\emph{Econometrica}, 55(3), 703--708.

\bibitem{phillips1988testing}
Phillips, P.~C.~B., \& Perron, P. (1988). Testing for a unit root in time series
regression. \emph{Biometrika}, 75(2), 335--346.

\bibitem{saiddickey1984}
Said, S.~E., \& Dickey, D.~A. (1984). Testing for unit roots in autoregressive-moving
average models of unknown order. \emph{Biometrika}, 71(3), 599--607.

\bibitem{stoikov2018microprice}
Stoikov, S. (2018). The micro-price: a high-frequency estimator of future prices.
\emph{Quantitative Finance}, 18(12), 1959--1966.

\bibitem{triantafyllopoulos2011dynamic}
Triantafyllopoulos, K., \& Montana, G. (2011). Dynamic modeling of mean-reverting
spreads for statistical arbitrage. \emph{Computational Management Science}, 8(1--2), 23--49.

\bibitem{urquhart2016inefficiency}
Urquhart, A. (2016). The inefficiency of Bitcoin. \emph{Economics Letters}, 148, 80--82.

\bibitem{vidyamurthy2004pairs}
Vidyamurthy, G. (2004). \emph{Pairs Trading: Quantitative Methods and Analysis}.
Wiley Finance.

\bibitem{wei2018liquidity}
Wei, W.~C. (2018). Liquidity and market efficiency in cryptocurrencies.
\emph{Economics Letters}, 168, 21--24.
```

Total: 4 + 19 = **23 entries.** (Counting Phillips-Perron, which the appendix invokes.)

---

## Section 2: Replacement Related Work Section

Replace the current `\section{Related work}` block (lines ~76--91 of `main.tex`) with the
following. ~700 words total across four subsections plus an opener and a "what is new" closer.

```latex
% =====================================================================
\section{Related work}
\label{sec:related}

Our work sits at the intersection of three literatures: the cointegration branch of pairs
trading, modern market microstructure, and empirical studies of cryptocurrency market
efficiency. The pairs literature provides the Kalman dynamic hedge and rolling $z$-score
screens whose artifacts we expose; the microstructure literature provides the order-flow
machinery we apply to crypto L2 data; and the crypto-efficiency literature is the context
in which our headline negative result lands.

\subsection{Pairs trading and dynamic cointegration}
\label{sec:related-pairs}

The cointegration branch of pairs trading descends from Engle and
Granger~\cite{engle1987cointegration}, with its practitioner-facing version codified in
Vidyamurthy~\cite{vidyamurthy2004pairs}: fit a spread, test the residual for stationarity,
trade deviations of the residual. The two screens we audit in Section~\ref{sec:artifacts}
are standard implementations of that pipeline. The state-space variant comes from Elliott,
van der Hoek, and Malcolm~\cite{elliott2005pairs}, who model the spread as a discrete-time
Ornstein-Uhlenbeck process observed in Gaussian noise. Triantafyllopoulos and
Montana~\cite{triantafyllopoulos2011dynamic} extend it to time-varying parameters with
online estimation, which is essentially the Kalman dynamic hedge we run. The distance-based
alternative of Gatev, Goetzmann, and Rouwenhorst~\cite{gatev2006pairs} is the standard
benchmark; Do and Faff~\cite{do2010simple} document its secular decay on US equities.
Harvey's classical state-space text~\cite{harvey1989structural} states the steady-state
whitening property we exploit, but the connection to downstream cointegration screening is
not drawn there.

\subsection{Microstructure and order flow}
\label{sec:related-micro}

The information content of order flow is the central object of classical microstructure,
beginning with Kyle~\cite{kyle1985continuous} on the linear price impact of informed
trading. Hasbrouck~\cite{hasbrouck1991measuring} introduces the permanent-versus-transient
decomposition of trade impact, which we use to interpret the seconds-scale half-life we
observe. The event-time order flow imbalance of Cont, Kukanov, and
Stoikov~\cite{contkukanovstoikov2014} predicts subsecond mid moves nearly linearly on
equities; we replicate this on Binance L2 in 2024 for BTC, ETH, SOL, and AVAX. We use the
size-weighted microprice of Stoikov~\cite{stoikov2018microprice} as our regression target.
The toxicity program of Easley, L\'opez de Prado, and O'Hara~\cite{easley2012vpin} is the
natural place to look for a tradable order-flow signal at longer horizons, but Andersen and
Bondarenko~\cite{andersen2014vpin} show its predictive content is largely mechanical and
its sign is sensitive to trade classification. Our crypto VPIN measurement replicates their
critique.

\subsection{Cryptocurrency market efficiency}
\label{sec:related-crypto}

The crypto efficiency literature has converged on a picture our results are consistent
with: the most-traded coins on the most-traded venues look weak-form efficient, while
thinner segments do not. Urquhart's~\cite{urquhart2016inefficiency} original Bitcoin tests
rejected EMH on early daily data but already noted the second half of the sample looked
tighter. Wei~\cite{wei2018liquidity} shows on 456 coins that return predictability and
Hurst persistence collapse with liquidity, and Brauneis and Mestel~\cite{brauneis2018price}
replicate the link across 73 coins and eight tests. Makarov and
Schoar~\cite{makarov2020trading} show that persistent cross-exchange price gaps survive
transaction costs and are best explained by capital-control and credit frictions, which
mirrors why our Coinbase-adjusted Sharpe sits below the Binance.US figure. The closest
prior on crypto pairs is Fil and Kristoufek~\cite{fil2020pairs}, who find that most
strategies underperform their benchmarks once costs are realistic; our honest-accounting
Sharpe of $\sim$$1.0$ is consistent with their picture.

\subsection{Machine learning approaches}
\label{sec:related-ml}

Krauss, Do, and Huck~\cite{krauss2017deep} apply deep neural networks, gradient-boosted
trees, and random forests to S\&P~500 statistical arbitrage and find ensemble lifts that
erode sharply after costs. Our XGBoost / LSTM / transformer comparison reproduces the same
pattern on crypto pairs: a calibrated spread $z$-score carries roughly 79\% of the P\&L of
a four-architecture stack, in line with the multiple-testing cautions of L\'opez de
Prado~\cite{lopezdeprado2018advances}. The deep-learning literature on limit-order-book
data predicts at sub-second horizons, which is shorter than our hourly bars and consistent
with our finding that L2 order-flow information decays within 30 seconds and does not
survive to a tradable horizon.

\subsection{What is new here}
\label{sec:related-novelty}

Three things in this paper are not in the literatures above. First, the matched-placebo
audit of two standard cointegration screens — the result that an ADF test on Kalman
innovations and a rolling $z$-score on a residual pass random-walk, phase-randomized, and
block-shuffled placebos at the same rate as real pairs, and the positive control showing
the Kalman screen cannot separate cointegrated from random-walk data — is, to our
knowledge, new. The underlying mechanisms (filter innovation whitening, rolling-mean
demeaning) are textbook; the placebo audit applied to the cointegration screen tradition
(\cite{vidyamurthy2004pairs, elliott2005pairs, triantafyllopoulos2011dynamic}) is not.
Second, the direct measurement of L2 execution value on crypto majors with 18{,}432
simulated parent orders, with a perfect-foresight oracle as the upper bound, is, to our
knowledge, the first such test for crypto. Third, the survivorship analysis on a
point-in-time universe that includes the LUNA, UST, and FTT collapses, with the
structural-break circuit breaker preserving roughly 96\% of monthly Sharpe at the 8\%/year
breakage rate consistent with historical delistings, is also new to crypto pair spreads.
```

---

## Section 3: Methods Appendix

Insert immediately before `\end{document}`. This is a verbatim drop-in. Only patches from
v1 are removal of references to `andrews1991hac`, `johansen1991estimation`, `hansen2005spa`,
and `politisromano1994stationary`, which were dropped from the bibliography.

```latex
\appendix

\section{Methods background}
\label{sec:appendix}

This appendix collects working definitions of the tools used in the body of the paper. It
is meant for a reader who has seen probability and basic time series but may not have used
all of these specific instruments. We keep proofs out and give the formulas that are
actually invoked in the text. Section references in parentheses point to where the tool is
used.

\subsection{Kalman filter and state-space form for dynamic hedge ratios}
\label{app:kalman}

Let $y_t$ be the dependent log price and $x_t$ the independent log price in a pair. The
dynamic hedge model treats the intercept $\alpha_t$ and slope $\beta_t$ as latent states
that drift over time. The state-space form is
\[
\theta_t \;=\; \theta_{t-1} + w_t, \qquad w_t \sim \mathcal{N}(0, Q),
\]
\[
y_t \;=\; H_t \, \theta_t + v_t, \qquad v_t \sim \mathcal{N}(0, R),
\]
with $\theta_t = (\alpha_t, \beta_t)^\top$, observation matrix $H_t = (1, x_t)$, state-noise
covariance $Q \in \mathbb{R}^{2\times 2}$, and observation-noise variance $R > 0$. The
states follow a random walk; only $y_t$ is observed.

Given the prior state mean $\hat{\theta}_{t-1|t-1}$ and covariance $P_{t-1|t-1}$, the
Kalman recursion predicts
\[
\hat{\theta}_{t|t-1} = \hat{\theta}_{t-1|t-1}, \qquad P_{t|t-1} = P_{t-1|t-1} + Q,
\]
forms the one-step innovation
\[
e_t \;=\; y_t - H_t \hat{\theta}_{t|t-1}, \qquad S_t \;=\; H_t P_{t|t-1} H_t^\top + R,
\]
computes the Kalman gain
\[
K_t \;=\; P_{t|t-1} H_t^\top S_t^{-1},
\]
and updates
\[
\hat{\theta}_{t|t} = \hat{\theta}_{t|t-1} + K_t e_t, \qquad P_{t|t} = (I - K_t H_t) P_{t|t-1}.
\]
The gain $K_t$ is the optimal blend of model prediction and new observation. The
hyperparameters $Q$ and $R$ are estimated by maximizing the Gaussian innovation likelihood
\[
\log \mathcal{L}(Q, R) \;=\; -\tfrac{1}{2} \sum_{t} \left( \log 2\pi S_t + e_t^2 / S_t \right),
\]
on a training window, then frozen and forward-rolled on the test window so no test-window
information leaks into the filter.

The mechanism behind the artifact in Section~\ref{sec:kalman} lives in the innovation
sequence. By construction, $\{e_t / \sqrt{S_t}\}$ has zero mean, unit variance, and zero
serial correlation under the model, and at steady state the filter approaches this
whitening behaviour for any input~\cite{harvey1989structural}. Running a stationarity test
on $e_t$ therefore measures how well the filter has been optimised, not whether $y_t$ and
$x_t$ share a common stochastic trend. A test that rejects the unit root on $e_t$ is
consistent with no cointegration whatsoever.

\subsection{Augmented Dickey-Fuller test}
\label{app:adf}

The augmented Dickey-Fuller (ADF) test of Said and Dickey~\cite{saiddickey1984} tests the
null $H_0$: the series $z_t$ has a unit root (is $I(1)$) against $H_1$: $z_t$ is
stationary. The test regression is
\[
\Delta z_t \;=\; \mu + \rho \, z_{t-1} + \sum_{i=1}^{p} \gamma_i \, \Delta z_{t-i} + \varepsilon_t,
\]
with the augmentation lags included to soak up short-run serial correlation in
$\varepsilon_t$. The test statistic is the $t$-ratio on $\hat{\rho}$, compared against
Dickey-Fuller critical values. Under $H_0$, $\rho = 0$ and the process is a random walk;
under $H_1$, $\rho < 0$ and shocks decay geometrically. Reject for sufficiently negative
$t_{\hat\rho}$. We supplement ADF with the nonparametric Phillips-Perron
test~\cite{phillips1988testing} for robustness.

The connection to the rolling-$z$ artifact (Section~\ref{sec:kalman}) is mechanical: a
series that is close to white noise has trivially fast mean reversion, so the ADF rejects
easily even when no economic mean-reverting relation is present. The test answers ``is this
stationary'', not ``is this a cointegrating residual''.

\subsection{Engle-Granger cointegration}
\label{app:eg}

Two integrated series $y_t, x_t \sim I(1)$ are cointegrated if there exists a constant
$\beta$ such that $y_t - \beta x_t$ is $I(0)$. The Engle-Granger two-step
procedure~\cite{engle1987cointegration} fits this $\beta$ by static OLS on levels,
\[
y_t \;=\; \alpha + \beta x_t + u_t,
\]
forms the residuals $\hat{u}_t = y_t - \hat\alpha - \hat\beta x_t$, and applies an ADF test
to $\hat{u}_t$ with Engle-Granger critical values (which are more conservative than ADF
because $\hat\beta$ has been estimated). Rejection is evidence that the linear combination
defined by $(\hat\alpha, \hat\beta)$ is a stationary error-correction term.

This is the clean comparison for the Kalman-innovations procedure of
Section~\ref{app:kalman}. Engle-Granger tests a residual whose dynamics are inherited from
the original prices; the Kalman procedure tests an innovation sequence whose dynamics are
inherited from the filter. The latter rejects the unit root on placebos that, by
construction, cannot be cointegrated; the former does not.

\subsection{Newey-West HAC standard errors}
\label{app:hac}

Ordinary OLS standard errors assume the regression errors are independent and
homoskedastic. In time series with overlapping returns, slow-moving regressors, or
autocorrelated targets, both assumptions fail and naive standard errors understate sampling
uncertainty, often by a large factor. The heteroskedasticity- and autocorrelation-consistent
(HAC) covariance estimator of Newey and West~\cite{newey1987hac} corrects this. For a
regression with score $g_t$, the Newey-West variance estimator is
\[
\hat{\Sigma}_{NW} \;=\; \hat{\Gamma}_0 + \sum_{\ell = 1}^{L} w_\ell \, \big( \hat{\Gamma}_\ell + \hat{\Gamma}_\ell^\top \big),
\qquad
w_\ell \;=\; 1 - \frac{\ell}{L+1},
\]
where $\hat{\Gamma}_\ell = T^{-1} \sum_{t = \ell+1}^{T} g_t g_{t-\ell}^\top$ is the sample
autocovariance at lag $\ell$ and $w_\ell$ is the Bartlett kernel. The kernel is
non-negative and linearly decreasing in $\ell$, which guarantees a positive semi-definite
estimate.

We use $L = 60$ for one-hour analyses with a 60-bar target, $L = 24$ for daily analyses
with a one-day target, and $L = 240$ for monthly Sharpe inference where ten daily lags
buffer the monthly aggregation. Sharpe ratios reported as ``HAC adjusted'' divide the mean
by the square root of the HAC variance of the mean, then scale to monthly.

\subsection{Deflated Sharpe Ratio}
\label{app:dsr}

The deflated Sharpe ratio (DSR) of Bailey and L\'opez de Prado~\cite{bailey2014dsr}
discounts a reported Sharpe for non-normal returns and for the number of strategies that
were tried before the one that was reported. Let $\widehat{SR}$ be the in-sample Sharpe of
the selected strategy, $T$ the sample length, $\gamma$ the sample skewness, and $\kappa$
the sample kurtosis. The DSR is
\[
\mathrm{DSR} \;=\; \Phi\!\left( \frac{ (\widehat{SR} - SR_0) \, \sqrt{T - 1} }
    { \sqrt{1 - \gamma \, \widehat{SR} + \tfrac{\kappa - 1}{4} \, \widehat{SR}^{\,2} } } \right),
\]
where $\Phi$ is the standard normal CDF. The threshold $SR_0$ is the expected maximum
Sharpe under the null of zero true edge across $N$ independent trials,
\[
SR_0 \;=\; \sqrt{V[\,\widehat{SR}_n\,]} \left( (1 - \gamma_E) \, \Phi^{-1}\!\big(1 - 1/N\big)
    + \gamma_E \, \Phi^{-1}\!\big(1 - 1/(N e)\big) \right),
\]
with $\gamma_E \approx 0.5772$ the Euler-Mascheroni constant and $V[\widehat{SR}_n]$ the
cross-trial variance of estimated Sharpes under the null.

$N$ matters because the maximum of $N$ noisy estimates grows with $N$: trying ten
strategies and reporting the best one has a much higher null Sharpe than trying one.
Doubling the number of trials raises $SR_0$ by roughly $\sqrt{\log N}$ in the relevant
regime, so a strategy that just beat a one-trial null can fail a hundred-trial null. We use
$N$ equal to the number of distinct parameter or pair configurations evaluated before the
reported one.

\subsection{Ornstein-Uhlenbeck process and reversion speed}
\label{app:ou}

The Ornstein-Uhlenbeck (OU) process is the canonical continuous-time mean-reverting model.
The SDE is
\[
dS_t \;=\; \kappa \, (\mu - S_t) \, dt + \sigma \, dW_t,
\]
with long-run mean $\mu$, reversion speed $\kappa > 0$, instantaneous volatility $\sigma$,
and $W_t$ a Wiener process. The conditional mean decays geometrically toward $\mu$ with
rate $\kappa$, so the expected time for a deviation to halve is the half-life
\[
t_{1/2} \;=\; \frac{\ln 2}{\kappa}.
\]
We estimate $\kappa$ from a discretized AR(1) fit on the spread,
$S_{t+1} - S_t = a + b (S_t - \mu) + \eta_t$ with $\hat{\kappa} = -\ln(1 + b) / \Delta t$,
and rank pairs by this estimate.

The finding in Section~\ref{sec:real} that train-window $\hat\kappa$ predicts test-window
$\hat\kappa$ at Spearman $\rho = 0.46$ across walk-forward splits is a statement about the
process, not about profitability.

\subsection{Walk-forward cross-validation}
\label{app:wf}

Walk-forward cross-validation is the non-anticipative analogue of $k$-fold CV for time
series. Fix a train length $T_{\mathrm{tr}}$, a test length $T_{\mathrm{te}}$, and a step
length $T_s$. Split $i$ uses observations $[t_i, t_i + T_{\mathrm{tr}})$ for fitting and
$[t_i + T_{\mathrm{tr}}, t_i + T_{\mathrm{tr}} + T_{\mathrm{te}})$ for evaluation, then
advances $t_{i+1} = t_i + T_s$. Any parameter (cointegrating $\beta$, OU $\kappa$, Kalman
$Q, R$, model weights) is refit per split using only the train window. Results are reported
as the distribution across splits, not as a single in-sample number.

We use $T_{\mathrm{tr}} = 6$ months, $T_{\mathrm{te}} = 3$ months, and $T_s = 3$ months in
most of the paper, with $90/30$ in earlier microstructure work. Random $k$-fold is
inappropriate for time series because it routinely places train observations after test
observations and destroys the no-lookahead guarantee.

\subsection{Placebo construction}
\label{app:placebo}

A placebo is a synthetic series with a known absence of the structure being tested.
Different nulls preserve different features, and the right placebo is the one that matches
everything in the data except the mechanism in question. We use four.

\emph{Random walks via cumulative sums.} Sample $\eta_t \sim \mathcal{N}(0, \sigma^2)$
i.i.d.~and set $z_t = z_{t-1} + \eta_t$. This preserves nothing about the original series
except the innovation variance; two independent random walks share no common trend.

\emph{Phase randomization.} Take the discrete Fourier transform $\tilde{z}_k =
\mathcal{F}[z_t]$, replace each non-DC phase by a uniform draw on $[0, 2\pi)$ subject to
Hermitian symmetry, and invert. The resulting series has the same power spectrum as the
original but random phase, so all linear second-order statistics are preserved while any
non-linear or cross-series alignment is destroyed.

\emph{Block shuffle.} Partition $z_t$ into contiguous blocks of length $L$ and concatenate
the blocks in a random order. Short-range autocorrelation up to lag $\sim L$ is preserved;
long-range structure including unit roots and cointegrating drift is destroyed.

\emph{Random-pair pairing.} Keep each price series intact but pair it with the price series
of an unrelated coin. Marginal distributions, autocorrelations, volatility clustering, and
trends are all preserved leg by leg; only the joint cointegration is destroyed.

The four nulls answer different questions. The Kalman screen of Section~\ref{sec:kalman}
fails the random-walk null and the random-pair null at the same rate as the real data,
which is the diagnostic.

\subsection{Microstructure terms}
\label{app:micro}

\emph{Mid quote.} $p^{\mathrm{mid}}_t = (p^a_t + p^b_t)/2$, with $p^a_t$ the best ask and
$p^b_t$ the best bid.

\emph{Quoted spread.} $s_t = p^a_t - p^b_t$. The relative spread is $s_t /
p^{\mathrm{mid}}_t$.

\emph{Microprice.} The size-weighted mid of Stoikov~\cite{stoikov2018microprice},
\[
p^{\mathrm{micro}}_t \;=\; \frac{q^b_t \, p^a_t + q^a_t \, p^b_t}{q^a_t + q^b_t},
\]
with $q^a_t, q^b_t$ the displayed sizes at the top of book. The microprice tilts toward the
side with less depth.

\emph{Order book imbalance.} For top $K$ levels,
\[
\mathrm{OBI}_t^{(K)} \;=\; \frac{\sum_{k=1}^{K} q^b_{t,k} - \sum_{k=1}^{K} q^a_{t,k}}
    {\sum_{k=1}^{K} q^b_{t,k} + \sum_{k=1}^{K} q^a_{t,k}} \;\in\; [-1, 1].
\]

\emph{Order flow imbalance.} The best-level event-time OFI of Cont, Kukanov, and
Stoikov~\cite{contkukanovstoikov2014} is built from changes in displayed depth at the inside,
\[
e_t \;=\; \mathbb{1}\{p^b_t \geq p^b_{t-1}\} q^b_t - \mathbb{1}\{p^b_t \leq p^b_{t-1}\} q^b_{t-1}
    - \mathbb{1}\{p^a_t \leq p^a_{t-1}\} q^a_t + \mathbb{1}\{p^a_t \geq p^a_{t-1}\} q^a_{t-1},
\]
and the aggregated OFI over a window is
$\mathrm{OFI}_{[\tau_1, \tau_2]} = \sum_{t: \tau_1 \leq t < \tau_2} e_t$. The $e_t$
contribution is positive for adds on the bid or lifts on the ask and negative for cancels
on the bid or hits on the bid.

\emph{VPIN.} The volume-synchronized probability of informed trading of Easley, L\'opez de
Prado, and O'Hara~\cite{easley2012vpin} groups trades into equal-volume buckets of size $V$
and computes, within bucket $j$,
\[
\mathrm{VPIN}_j \;=\; \frac{1}{n} \sum_{i=j-n+1}^{j} \frac{|V_i^B - V_i^S|}{V},
\]
with $V_i^B$ and $V_i^S$ the buy- and sell-classified volume in bucket $i$ and $n$ the
window length in buckets.

\emph{Permanent vs transient impact.} Hasbrouck~\cite{hasbrouck1991measuring} decomposes
the mid-quote response to a trade into a permanent component (information) that persists in
the efficient price and a transient component (liquidity) that decays as inventory works
off.

\subsection{Block bootstrap}
\label{app:bb}

The i.i.d.~bootstrap is invalid for autocorrelated series because it destroys the serial
structure that drives sampling uncertainty. The block bootstrap of
K\"unsch~\cite{kunsch1989block} preserves it. From a series $\{z_t\}_{t=1}^{T}$, draw
$\lceil T / L \rceil$ blocks of length $L$ with replacement, where each block is a
contiguous slice $(z_s, z_{s+1}, \dots, z_{s+L-1})$ with $s$ uniform on
$\{1, \dots, T - L + 1\}$, concatenate them, and truncate to length $T$. Resample $B$ times
to form the bootstrap distribution.

The block length $L$ governs the bias-variance trade-off: $L$ must be large enough that
adjacent blocks are approximately independent, but small enough that enough distinct blocks
exist. We use $L = 300$ for hourly series, roughly twelve trading days. Use the block
bootstrap whenever the statistic is sensitive to serial correlation, including Sharpe
ratios on returns and the mean of overlapping forecasts.
```

---

## Section 4: Figure Specifications

Numbered in priority order (must-have, should-have, nice-to-have). For each, the data
source, the section to insert into, and a runnable sketch are given. Unchanged from v1.

### Figure 1 (must-have) — `fig_kalman_innovations_white.pdf`

- **Caption:** A Kalman filter whitens its own innovations: representative real pair (top) and an independent random walk pair (bottom) produce visually indistinguishable innovation series, with sample ACFs flat past lag 1.
- **Section:** `sec:kalman` — visual companion to the placebo tables.
- **Data source:** Re-run `scratch/kalman_positive_control.py` or `scratch/audit_part1.py` to dump per-bar innovations for one real pair (e.g. BTCUSDT_ETHUSDT) and one independent-RW placebo.
- **Sketch:**
```python
fig, axes = plt.subplots(2, 2, figsize=(8, 5), sharex='col')
for i, (label, innov) in enumerate([('Real BTC/ETH', real_innov),
                                    ('Indep. RW placebo', rw_innov)]):
    axes[i,0].plot(innov, lw=0.5, color='C0' if i==0 else 'C3')
    axes[i,0].set_title(f'{label} - Kalman innovations')
    acf = sm.tsa.acf(innov, nlags=40)
    axes[i,1].stem(acf, basefmt=' ')
    axes[i,1].axhline(1.96/len(innov)**0.5, ls='--', color='gray')
    axes[i,1].set_title('Sample ACF')
plt.tight_layout(); plt.savefig('fig_kalman_innovations_white.pdf')
```

### Figure 2 (must-have) — `fig_poscontrol_vs_negcontrol.pdf`

- **Caption:** Positive vs negative control: the Kalman screen passes both the truly-cointegrated pairs and independent random walks at near 100%, while the clean Engle-Granger and static OLS tests separate the two groups.
- **Section:** `sec:kalman`.
- **Data source:** `scratch/kalman_positive_control.csv`.
- **Sketch:**
```python
df = pd.read_csv('scratch/kalman_positive_control.csv')
tests = ['kalman_adf_pass', 'clean_eg_pass', 'static_adf_pass']
labels = ['Kalman ADF', 'Engle-Granger (clean)', 'Static OLS ADF (clean)']
x = np.arange(len(tests)); w = 0.35
fig, ax = plt.subplots(figsize=(7,4))
ax.bar(x-w/2, df.loc[df.group.str.startswith('POS'), tests].values[0], w,
       label='Cointegrated (positive control)', color='C2')
ax.bar(x+w/2, df.loc[df.group.str.startswith('NEG'), tests].values[0], w,
       label='Independent RW (negative control)', color='C3')
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel('Pass rate (%)')
ax.legend(); plt.tight_layout(); plt.savefig('fig_poscontrol_vs_negcontrol.pdf')
```

### Figure 3 (must-have) — `fig_reversion_persistence_scatter.pdf`

- **Caption:** In-sample OU reversion speed predicts out-of-sample excess reversion across 9 walk-forward splits (Spearman $\rho = 0.46$); the rank-shuffled placebo (right) collapses to noise.
- **Section:** `sec:real`.
- **Data source:** `scratch/persistence_pairs.csv`.

### Figure 4 (must-have) — `fig_circuit_breaker_retention.pdf`

- **Caption:** A structural-break circuit breaker preserves the no-stop reversion Sharpe across injected delisting rates; without it, the Sharpe collapses past 10%/year break rate.
- **Section:** `sec:survivor`.
- **Data source:** `scratch/survivorship_adjusted_sharpe.csv`.

### Figure 5 (must-have) — `fig_forced_collapse_perpair.pdf`

- **Caption:** Per-pair P\&L when the no-stop rule is forced to hold three real delisting collapses (LUNA, UST, FTT), with and without the structural-break circuit breaker.
- **Section:** `sec:survivor`.
- **Data source:** `scratch/forced_collapse.csv`.

### Figure 6 (should-have) — `fig_ofi_decay.pdf`

- **Caption:** Predictive incremental $R^2$ of book-OFI for forward midprice moves decays from $\sim$10--15\% contemporaneously to near zero by 30 seconds.
- **Section:** `sec:real`.
- **Data source:** Re-run `scratch/book_ofi_incremental.py` with `horizons = [1, 2, 5, 10, 30, 60, 120, 300]`.

### Figure 7 (should-have) — `fig_microcap_adv.pdf`

- **Caption:** Median daily USD volume on Binance.US: the preregistered held-out microcaps trade at hundreds to a few thousand dollars per day, three to four orders of magnitude below the majors whose costs the backtest assumes.
- **Section:** `sec:survivor`.
- **Data source:** `scratch/microcap_adv.csv`.

### Figure 8 (should-have) — `fig_hac_inflation.pdf`

- **Caption:** Hourly Sharpe is heavily serial-correlation-inflated by multi-week holds: per-split IID vs Newey-West HAC ratios cluster around 1.5--2.5x across 27 walk-forward splits.
- **Section:** `sec:survivor`.
- **Data source:** `docs/hac_sharpe_per_split.csv`.

### Figure 9 (nice-to-have) — `fig_cancellation_share.pdf`

- **Caption:** Cancellations dominate trades in best-level liquidity withdrawal: 81--85\% of best-level size reductions are cancels, invisible to trade-only microstructure analysis.
- **Section:** `sec:real`.
- **Data source:** `scratch/book_ofi_2024.log` cancel-share percentages.

### Existing figure: reshape `fig_freq_invariance.pdf`

Do NOT cut it. Mihai's complaint is accurate as-drawn (it duplicates Table 3 percentages
with bars), but the underlying point — that real and placebo pass at identical rates across
hourly/4h/daily — is the visual punchline of the Kalman artifact section. Reshape to show
the **placebo gap** rather than the levels:

- For each frequency, plot `(real - placebo)` Kalman pass-rate gap at $p<0.05$ and
  $p<0.001$, alongside `(real - placebo)` for the clean Engle-Granger test.
- The Kalman screen's gap is essentially zero at every frequency (the artifact); the
  clean-test gap is large. A flat zero line for the Kalman delta vs. a tall bar for the
  clean delta lands the message Table 3 cannot.

Keep `fig_rollingz_noise.pdf` as is.

---

## Section 5: Table Specifications

### Table T1 (must-have) — `tab:venue-comparison`

- **Caption:** Naive vs HAC-adjusted monthly Sharpe on Binance.US and Coinbase for the no-stop reversion strategy, showing the venue gap and the serial-correlation adjustment.
- **Section:** `sec:survivor` (Section 6.3).
- **Data source:** `docs/hac_sharpe_fixed.csv` and `docs/hac_sharpe_fixed_deep.csv`.

### Table T2 (should-have) — `tab:exec-symbol-breakdown`

- **Caption:** Execution costs by symbol and order size show aggressive crossing dominates for all four majors at both \$10k and \$50k notionals; the L3-from-L2 signal never beats the random-rate placebo.
- **Section:** `sec:real`.
- **Data source:** `scratch/exec_value_2024_summary.csv`.

### Table T3 (should-have) — `tab:institutional-impact`

- **Caption:** Per-trade price impact at 1-second horizon by trade size bucket and symbol: institutional ($>$\$10k) consistently 2.0--2.6x retail across BTC/ETH/SOL/AVAX.
- **Section:** `sec:real`.
- **Data source:** `scratch/impact_decomp_2024.log`.

---

## Section 6: Replacement Contributions Paragraph

Replace the current "We make three contributions" paragraph (lines ~56--68 of `main.tex`)
with the following. Honest about what is new vs textbook vs replication.

```latex
We make three contributions, and we are explicit about which are new and which are
replication. First, two standard pair-screening tools manufacture the appearance of
structure on data that cannot carry it. A Kalman dynamic hedge cointegration test passes
random-walk, phase-randomized, and block-shuffled placebos at the same near-$100\%$ rate it
passes real pairs, and a positive control of constructed cointegrated pairs passes at the
same rate as the random-walk negative: the screen carries no information about
cointegration. A rolling $z$-score reverts mechanically on pure random walks, producing
per-event reversion of $+0.98z$ to $+1.61z$ across hourly, four-hour, and daily frequencies
and a Sharpe near $2.2$ on random-pair placebos against $4.75$ on the actual data. The
underlying mechanisms — that a well-specified Kalman filter has white steady-state
innovations~\cite{harvey1989structural} and that rolling-mean demeaning induces negative
autocorrelation — are textbook. The matched-placebo demonstration applied to the
cointegration screen tradition~\cite{vidyamurthy2004pairs, elliott2005pairs,
triantafyllopoulos2011dynamic} is, to our knowledge, new. Second, we measure whether L2
order book signals improve execution placement for crypto majors with $18{,}432$ simulated
parent orders on event-level Binance L2 data. They do not: aggressive crossing is cheapest
across all four symbols, the signal correlates $+0.06$ with the per-order post-vs-cross
advantage, and a perfect-foresight oracle would save $1.6$ bps so the opportunity exists
but no contemporaneous feature forecasts it. Third, the candidate market-neutral reversion
strategy that emerges from the surviving structure clears honest accounting at a venue- and
HAC-adjusted monthly Sharpe near $1.0$, only without a stop, and a preregistered held-out
backtest of it inflates above the in-sample number on uncostable microcap moves; our one
surviving result ends up cutting against itself. The takeaway is the methodological one:
trust a cointegration or mean-reversion screen only after it fails on a placebo built to
reproduce the claim if the claim is mechanical.
```

---

## Section 7: Priority Order

1. **Bibliography merge (Section 1).** Add the 19 new `\bibitem` entries. Keep the four
   already in the paper. Verify no duplicate keys. **Blocker for:** every later section.
2. **Replace the contributions paragraph (Section 6)** in `\section{Introduction}`. Quick
   win — do this before Related Work so the intro frames the new related-work scope.
3. **Replace `\section{Related work}` (Section 2)** with the four-subsection version.
4. **Append Methods Appendix (Section 3)** before `\end{document}`. Verify `\appendix` is
   not already declared earlier.
5. **Add Table T1** (must-have, plain LaTeX).
6. **Generate must-have figures (Section 4, Figures 1--5).** Data already in repo:
   - Fig 1, 2: rerun `scratch/kalman_positive_control.py` if innovation series aren't
     cached.
   - Fig 3: `scratch/persistence_pairs.csv` ready.
   - Fig 4: `scratch/survivorship_adjusted_sharpe.csv` ready.
   - Fig 5: `scratch/forced_collapse.csv` ready.
7. **Reshape `fig_freq_invariance.pdf`** per the note in Section 4.
8. **Generate should-have figures and tables (Figures 6--8, Tables T2--T3).**
9. **Generate Figure 9 (nice-to-have)** if time permits.
10. **Title.** Still `TBD`. Suggested: *"Screening artifacts in cryptocurrency pair spreads:
    why Kalman cointegration tests pass random walks, and what survives once they don't"*.
11. **Final pass.** Recompile and check: every `\cite{}` resolves, `\label{}` / `\ref{}`
    consistency, figure file names match `\includegraphics{}`, word count.

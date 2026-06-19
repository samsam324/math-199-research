# Paper Revision Tasks

## Section 0: Summary

This document compiles all revisions needed for `paper/main.tex` based on Mihai's feedback (thin background, two figures one of which duplicates a table, unclear novelty, paper too short for a three-student project, need self-contained methods for non-experts). It contains: (1) the merged bibliography additions; (2) a full replacement `\section{Related work}` (~1500 words, four subsections); (3) a complete methods appendix (A.1 through A.10); (4) nine new figure specifications with data sources and Python sketches, plus instructions to reshape `fig_freq_invariance.pdf`; (5) three new table specifications; (6) a rewritten contributions paragraph for the introduction; (7) a priority order for applying changes. Apply in the order given in Section 7. All `\cite{}` keys in the new prose match the bibitems in Section 1, so the bibliography must be merged first or `\bibitem` lookups will fail at compile.

---

## Section 1: New Bibliography Entries

### Keep these from current paper

- `bailey2014dsr` (cited in intro and in the new ML / methods appendix material — keep)
- `engle1987cointegration` (cited in current related work and the new Pairs subsection — keep)
- `fil2020pairs` (cited in current related work and new Crypto subsection — keep)
- `gatev2006pairs` (cited in intro and new Pairs subsection — keep)

### Add these (deduplicated, sorted by key)

When two agents cited the same work, only one canonical entry appears below. Note the following dedup choices:

- `newey1987hac`: kept once (cited by Pairs and Methods agents).
- `andrews1991hac`: kept once.
- `johansen1991estimation`: kept once (Methods agent used key `johansen1991cointegration`; canonical key here is `johansen1991estimation` — make sure the Methods appendix uses this key).
- `saidDickey1984`: kept once (Methods agent used `saiddickey1984adf`; canonical key here is `saidDickey1984` — appendix uses `saiddickey1984` lowercased; either is fine, **standardize on `saiddickey1984`** to match the appendix draft).
- `fil2020pairs` already in paper — do NOT add again.
- `engle1987cointegration` already in paper — do NOT add again.
- The Crypto-efficiency agent reused `fil2020pairs` — already covered.

```latex
\bibitem{aleti2021bitcoin} Aleti, S., \& Mizrach, B. (2021). Bitcoin spot and futures market microstructure. \emph{Journal of Futures Markets}, 41(2), 194--225.

\bibitem{almgren2000optimal} Almgren, R., \& Chriss, N. (2000). Optimal execution of portfolio transactions. \emph{Journal of Risk}, 3(2), 5--39.

\bibitem{andersen2014vpin} Andersen, T.~G., \& Bondarenko, O. (2014). VPIN and the flash crash. \emph{Journal of Financial Markets}, 17, 1--46.

\bibitem{andrews1991hac} Andrews, D.~W.~K. (1991). Heteroskedasticity and autocorrelation consistent covariance matrix estimation. \emph{Econometrica}, 59(3), 817--858.

\bibitem{avellaneda2010statarb} Avellaneda, M., \& Lee, J.-H. (2010). Statistical arbitrage in the US equities market. \emph{Quantitative Finance}, 10(7), 761--782.

\bibitem{borovkova2019ensemble} Borovkova, S., \& Tsiamas, I. (2019). An ensemble of LSTM neural networks for high-frequency stock market classification. \emph{Journal of Forecasting}, 38(6), 600--619.

\bibitem{brandvold2015price} Brandvold, M., Moln\'ar, P., Vagstad, K., \& Valstad, O.~C.~A. (2015). Price discovery on Bitcoin exchanges. \emph{Journal of International Financial Markets, Institutions and Money}, 36, 18--35.

\bibitem{brauneis2018price} Brauneis, A., \& Mestel, R. (2018). Price discovery of cryptocurrencies: Bitcoin and beyond. \emph{Economics Letters}, 165, 58--61.

\bibitem{brogaard2014hft} Brogaard, J., Hendershott, T., \& Riordan, R. (2014). High-frequency trading and price discovery. \emph{Review of Financial Studies}, 27(8), 2267--2306.

\bibitem{caldeira2013selection} Caldeira, J., \& Moura, G.~V. (2013). Selection of a portfolio of pairs based on cointegration: A statistical arbitrage strategy. \emph{Brazilian Review of Finance}, 11(1), 49--80.

\bibitem{caporale2018persistence} Caporale, G.~M., Gil-Alana, L., \& Plastun, A. (2018). Persistence in the cryptocurrency market. \emph{Research in International Business and Finance}, 46, 141--148.

\bibitem{cartea2015algo} Cartea, \'A., Jaimungal, S., \& Penalva, J. (2015). \emph{Algorithmic and High-Frequency Trading}. Cambridge University Press.

\bibitem{contkukanovstoikov2014} Cont, R., Kukanov, A., \& Stoikov, S. (2014). The price impact of order book events. \emph{Journal of Financial Econometrics}, 12(1), 47--88.

\bibitem{cont2017optimal} Cont, R., \& Kukanov, A. (2017). Optimal order placement in limit order markets. \emph{Quantitative Finance}, 17(1), 21--39.

\bibitem{dixon2020mlfinance} Dixon, M.~F., Halperin, I., \& Bilokon, P. (2020). \emph{Machine Learning in Finance: From Theory to Practice}. Springer, Cham.

\bibitem{do2010simple} Do, B., \& Faff, R. (2010). Does simple pairs trading still work? \emph{Financial Analysts Journal}, 66(4), 83--95.

\bibitem{easley1987price} Easley, D., \& O'Hara, M. (1987). Price, trade size, and information in securities markets. \emph{Journal of Financial Economics}, 19(1), 69--90.

\bibitem{easley1996liquidity} Easley, D., Kiefer, N.~M., O'Hara, M., \& Paperman, J.~B. (1996). Liquidity, information, and infrequently traded stocks. \emph{Journal of Finance}, 51(4), 1405--1436.

\bibitem{easley2012vpin} Easley, D., L\'opez de Prado, M.~M., \& O'Hara, M. (2012). Flow toxicity and liquidity in a high-frequency world. \emph{Review of Financial Studies}, 25(5), 1457--1493.

\bibitem{easley2019mining} Easley, D., O'Hara, M., \& Basu, S. (2019). From mining to markets: The evolution of bitcoin transaction fees. \emph{Journal of Financial Economics}, 134(1), 91--109.

\bibitem{elliott2005pairs} Elliott, R.~J., van der Hoek, J., \& Malcolm, W.~P. (2005). Pairs trading. \emph{Quantitative Finance}, 5(3), 271--276.

\bibitem{fischer2018deep} Fischer, T., \& Krauss, C. (2018). Deep learning with long short-term memory networks for financial market predictions. \emph{European Journal of Operational Research}, 270(2), 654--669.

\bibitem{glosten1985bidask} Glosten, L.~R., \& Milgrom, P.~R. (1985). Bid, ask and transaction prices in a specialist market with heterogeneously informed traders. \emph{Journal of Financial Economics}, 14(1), 71--100.

\bibitem{hansen2005spa} Hansen, P.~R. (2005). A test for superior predictive ability. \emph{Journal of Business \& Economic Statistics}, 23(4), 365--380.

\bibitem{harvey1989structural} Harvey, A.~C. (1989). \emph{Forecasting, Structural Time Series Models and the Kalman Filter}. Cambridge University Press.

\bibitem{harvey2016crosssection} Harvey, C.~R., Liu, Y., \& Zhu, H. (2016). $\ldots$ and the cross-section of expected returns. \emph{Review of Financial Studies}, 29(1), 5--68.

\bibitem{hasbrouck1991measuring} Hasbrouck, J. (1991). Measuring the information content of stock trades. \emph{Journal of Finance}, 46(1), 179--207.

\bibitem{hasbrouck2007empirical} Hasbrouck, J. (2007). \emph{Empirical Market Microstructure: The Institutions, Economics, and Econometrics of Securities Trading}. Oxford University Press.

\bibitem{huang2015queue} Huang, W., Lehalle, C.-A., \& Rosenbaum, M. (2015). Simulating and analyzing order book data: the queue-reactive model. \emph{Journal of the American Statistical Association}, 110(509), 107--122.

\bibitem{johansen1991estimation} Johansen, S. (1991). Estimation and hypothesis testing of cointegration vectors in Gaussian vector autoregressive models. \emph{Econometrica}, 59(6), 1551--1580.

\bibitem{krauss2017deep} Krauss, C., Do, X.~A., \& Huck, N. (2017). Deep neural networks, gradient-boosted trees, random forests: statistical arbitrage on the S\&P~500. \emph{European Journal of Operational Research}, 259(2), 689--702.

\bibitem{krauss2017survey} Krauss, C. (2017). Statistical arbitrage pairs trading strategies: Review and outlook. \emph{Journal of Economic Surveys}, 31(2), 513--545.

\bibitem{kristoufek2018bitcoin} Kristoufek, L. (2018). On Bitcoin markets (in)efficiency and its evolution. \emph{Physica A: Statistical Mechanics and its Applications}, 503, 257--262.

\bibitem{kunsch1989block} K\"unsch, H.~R. (1989). The jackknife and the bootstrap for general stationary observations. \emph{Annals of Statistics}, 17(3), 1217--1241.

\bibitem{kyle1985continuous} Kyle, A.~S. (1985). Continuous auctions and insider trading. \emph{Econometrica}, 53(6), 1315--1335.

\bibitem{leung2019constructing} Leung, T., \& Nguyen, H. (2019). Constructing cointegrated cryptocurrency portfolios for statistical arbitrage. \emph{Studies in Economics and Finance}, 36(4), 581--599.

\bibitem{lintilhac2017model} Lintilhac, P.~S., \& Tourin, A. (2017). Model-based pairs trading in the bitcoin markets. \emph{Quantitative Finance}, 17(5), 703--716.

\bibitem{lopezdeprado2018advances} L\'opez de Prado, M. (2018). \emph{Advances in Financial Machine Learning}. John Wiley \& Sons, Hoboken, NJ.

\bibitem{makarov2020trading} Makarov, I., \& Schoar, A. (2020). Trading and arbitrage in cryptocurrency markets. \emph{Journal of Financial Economics}, 135(2), 293--319.

\bibitem{newey1987hac} Newey, W.~K., \& West, K.~D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. \emph{Econometrica}, 55(3), 703--708.

\bibitem{ohara2015hft} O'Hara, M. (2015). High frequency market microstructure. \emph{Journal of Financial Economics}, 116(2), 257--270.

\bibitem{phillips1988testing} Phillips, P.~C.~B., \& Perron, P. (1988). Testing for a unit root in time series regression. \emph{Biometrika}, 75(2), 335--346.

\bibitem{politisromano1994stationary} Politis, D.~N., \& Romano, J.~P. (1994). The stationary bootstrap. \emph{Journal of the American Statistical Association}, 89(428), 1303--1313.

\bibitem{saiddickey1984} Said, S.~E., \& Dickey, D.~A. (1984). Testing for unit roots in autoregressive-moving average models of unknown order. \emph{Biometrika}, 71(3), 599--607.

\bibitem{sensoy2019inefficiency} Sensoy, A. (2019). The inefficiency of Bitcoin revisited: A high-frequency analysis with alternative currencies. \emph{Finance Research Letters}, 28, 68--73.

\bibitem{sirignano2019deep} Sirignano, J.~A. (2019). Deep learning for limit order books. \emph{Quantitative Finance}, 19(4), 549--570.

\bibitem{sirignano2019universal} Sirignano, J., \& Cont, R. (2019). Universal features of price formation in financial markets: perspectives from deep learning. \emph{Quantitative Finance}, 19(9), 1449--1459.

\bibitem{stoikov2018microprice} Stoikov, S. (2018). The micro-price: a high-frequency estimator of future prices. \emph{Quantitative Finance}, 18(12), 1959--1966.

\bibitem{tran2020efficiency} Tran, V.~L., \& Leirvik, T. (2020). Efficiency in the markets of crypto-currencies. \emph{Finance Research Letters}, 35, 101382.

\bibitem{triantafyllopoulos2011dynamic} Triantafyllopoulos, K., \& Montana, G. (2011). Dynamic modeling of mean-reverting spreads for statistical arbitrage. \emph{Computational Management Science}, 8(1--2), 23--49.

\bibitem{urquhart2016inefficiency} Urquhart, A. (2016). The inefficiency of Bitcoin. \emph{Economics Letters}, 148, 80--82.

\bibitem{vidaltomas2018semistrong} Vidal-Tom\'as, D., \& Ib\'a\~nez, A. (2018). Semi-strong efficiency of Bitcoin. \emph{Finance Research Letters}, 27, 259--265.

\bibitem{vidyamurthy2004pairs} Vidyamurthy, G. (2004). \emph{Pairs Trading: Quantitative Methods and Analysis}. Wiley Finance, Hoboken.

\bibitem{wei2018liquidity} Wei, W.~C. (2018). Liquidity and market efficiency in cryptocurrencies. \emph{Economics Letters}, 168, 21--24.

\bibitem{zhang2019deeplob} Zhang, Z., Zohren, S., \& Roberts, S. (2019). DeepLOB: deep convolutional neural networks for limit order books. \emph{IEEE Transactions on Signal Processing}, 67(11), 3001--3012.
```

**Key consistency notes for the Windows agent:**

- The microstructure subsection draft uses `\cite{cont2014priceimpact}` but I've standardized the key as `contkukanovstoikov2014` (matches the appendix). If you keep the draft text as-is, change the cite key to `contkukanovstoikov2014`.
- `easley2012flow` (microstructure agent) and `easley2012vpin` (appendix) are the same paper — use `easley2012vpin`.
- `hasbrouck1991information` (microstructure agent) and `hasbrouck1991measuring` (appendix) are the same paper — use `hasbrouck1991measuring`.
- `saidDickey1984` / `saiddickey1984adf` / `saiddickey1984` — use `saiddickey1984` (all lowercase, no suffix).
- `johansen1991estimation` / `johansen1991cointegration` — use `johansen1991estimation`.

---

## Section 2: Replacement Related Work Section

Replace the current `\section{Related work}` block (lines 76--91 of `main.tex`) with the following.

```latex
% =====================================================================
\section{Related work}
\label{sec:related}

Our work sits at the intersection of four literatures: classical and modern market
microstructure, the cointegration branch of pairs trading and statistical arbitrage,
empirical studies of cryptocurrency efficiency, and machine learning for financial time
series. We use tools from each, and our findings interact with each one differently. The
microstructure literature furnishes the order-flow machinery we apply to crypto L2 data;
the pairs literature furnishes the Kalman dynamic hedge and rolling $z$-score screens whose
artifacts we expose; the crypto-efficiency literature is where our headline negative result
on majors lands; and the ML-for-finance literature provides the multiple-testing discipline
that frames our ablation. We summarize what each tradition assumes, what is known, and what
is left for us to add.

\subsection{Microstructure and order flow}
\label{sec:related-micro}

The information content of order flow is the central object of classical microstructure.
Kyle~\cite{kyle1985continuous} and Glosten and Milgrom~\cite{glosten1985bidask} derive the
linear price impact of informed orders and the adverse-selection component of the spread,
and Easley and O'Hara~\cite{easley1987price} extend the result to trade size. Hasbrouck's
VAR decomposition~\cite{hasbrouck1991measuring} turns these ideas into an estimator that
separates permanent from transient price moves, and the textbook treatments of
Hasbrouck~\cite{hasbrouck2007empirical} and Cartea, Jaimungal, and
Penalva~\cite{cartea2015algo} cover the empirical machinery we use. Our incremental-$R^2$
attribution between book-driven and trade-driven flow sits inside this tradition.

At the event level we use the order flow imbalance of Cont, Kukanov, and
Stoikov~\cite{contkukanovstoikov2014}, who show on NYSE TAQ that the net of additions,
cancellations, and trades at the best bid and ask explains contemporaneous mid-price changes
nearly linearly. We find the same on Binance.com L2 in 2024 for BTC, ETH, SOL, and AVAX. The
dependent variable in those regressions is the microprice of
Stoikov~\cite{stoikov2018microprice}, a size-weighted mid that absorbs the mechanical
component of imbalance. Cancellations dominate withdrawals at the top of the book in our
data, consistent with the queue-reactive limit order book of Huang, Lehalle, and
Rosenbaum~\cite{huang2015queue}. The toxicity program is the natural place to look for a
tradable order-flow signal at longer horizons. Easley, L\'opez de Prado, and
O'Hara~\cite{easley1996liquidity, easley2012vpin} propose PIN and VPIN for that purpose, but
Andersen and Bondarenko~\cite{andersen2014vpin} show that VPIN's predictive content is
largely mechanical and that its sign is sensitive to the trade classifier. We re-derive
VPIN on crypto and find it inverts, behaving as a directional-consensus meter rather than a
toxicity meter, in line with their critique.

Whether any of this is exploitable depends on execution. Almgren and
Chriss~\cite{almgren2000optimal} frame the problem as impact against timing risk, and Cont
and Kukanov~\cite{cont2017optimal} formalize the limit-versus-market choice conditional on
queue size and fees. O'Hara~\cite{ohara2015hft} and Brogaard, Hendershott, and
Riordan~\cite{brogaard2014hft} argue that at sub-second horizons the surplus is captured by
fast participants and that what reaches a slower trader is mostly transient. Our negative
execution-value result for simulated passive-versus-cross decisions on majors sits inside
these constraints: the contemporaneous signal is real, the 30-second decay is consistent
with the HFT-era picture, and the residual is too small to fund the cost.

\subsection{Pairs trading and dynamic cointegration}
\label{sec:related-pairs}

The pairs trading literature splits, following Krauss~\cite{krauss2017survey}, into a
distance branch descended from Gatev, Goetzmann, and Rouwenhorst~\cite{gatev2006pairs} and a
cointegration branch descended from Engle and Granger~\cite{engle1987cointegration} and
Johansen~\cite{johansen1991estimation}. The cointegration branch is the one we work in. Its
practitioner-facing version is Vidyamurthy~\cite{vidyamurthy2004pairs}, who codifies the
standard pipeline of unit-root testing the residual of a fitted spread and trading the
deviations of that residual when it strays. Two of our screens, an augmented Dickey-Fuller
test~\cite{saiddickey1984, phillips1988testing} on a spread and a rolling $z$-score on the
residual, are canonical implementations of that pipeline.

The state-space variant we audit comes from Elliott, van der Hoek, and
Malcolm~\cite{elliott2005pairs}, who model the spread as a discrete-time Ornstein-Uhlenbeck
process observed in Gaussian noise and estimate it with the Kalman filter. Triantafyllopoulos
and Montana~\cite{triantafyllopoulos2011dynamic} extend this to time-varying parameters with
on-line estimation, which is essentially the Kalman dynamic hedge we run in
Section~\ref{sec:kalman}. The broader stat-arb literature in the same idiom, including
Avellaneda and Lee~\cite{avellaneda2010statarb} on US equities and Caldeira and
Moura~\cite{caldeira2013selection} on emerging-market portfolios, all share the same move:
build a mean-reverting residual, test it for stationarity in sample, trade the reversion.
None of these works tests the screen itself against a placebo that cannot be cointegrated.
Harvey's classical state-space text~\cite{harvey1989structural} states the relevant fact
directly, that a correctly specified Kalman filter at steady state produces white
innovations, but the connection to downstream cointegration screening is not drawn.

The closest profitability results are Do and Faff~\cite{do2010simple}, who document the
secular decay of simple pairs trading on US equities outside of turbulent regimes. The
methodological literature on HAC inference~\cite{newey1987hac, andrews1991hac} provides the
right precedent: a screen that looks decisive can collapse once the right correction is
applied.

\subsection{Cryptocurrency market efficiency}
\label{sec:related-crypto}

The crypto efficiency literature has converged on a picture our results are consistent with:
the most-traded coins on the most-traded venues look weak-form efficient, while thinner
segments do not. Urquhart's~\cite{urquhart2016inefficiency} original Bitcoin tests rejected
EMH on 2010--2016 daily data but already noted the second half of the sample looked tighter,
and Kristoufek's~\cite{kristoufek2018bitcoin} Efficiency Index sees the same pattern with
efficient windows clustered after bubble cooldowns. Caporale, Gil-Alana, and
Plastun~\cite{caporale2018persistence} report long memory on four early majors and trend-rule
abnormal profit through 2017. Reruns on later data point the other way.
Sensoy~\cite{sensoy2019inefficiency} shows BTC/USD and BTC/EUR efficiency improving sharply
after 2016 in high-frequency entropy tests, Tran and Leirvik~\cite{tran2020efficiency}
document the time-varying weak-form efficiency of the five largest coins increasing with
volume and volatility, and Vidal-Tom\'as and Ib\'a\~nez~\cite{vidaltomas2018semistrong} push
the conclusion to semi-strong form on Bitstamp event studies. Liquidity is the variable doing
the work in both directions: Wei~\cite{wei2018liquidity} shows on 456 coins that return
predictability and Hurst persistence collapse with liquidity, and Brauneis and
Mestel~\cite{brauneis2018price} replicate the same liquidity-efficiency link across 73 coins
and eight tests.

The microstructure literature fills in why. Brandvold et al.~\cite{brandvold2015price}
establish that BTC price discovery concentrates in the most active venues with shares that
drift over time, Aleti and Mizrach~\cite{aleti2021bitcoin} extend the same Hasbrouck
machinery to CME futures and spot, and Easley, O'Hara, and Basu~\cite{easley2019mining}
document the emergence of binding transaction-fee dynamics on the blockchain itself. Makarov
and Schoar~\cite{makarov2020trading} show that persistent cross-exchange price gaps survive
transaction costs and are best explained by capital-control and credit frictions rather than
informational inefficiency, which mirrors why our Coinbase-adjusted Sharpe sits below the
Binance.US figure.

The pairs-trading thread on crypto is shorter. Lintilhac and Tourin~\cite{lintilhac2017model}
solve a control problem on cointegrated BTC portfolios across three exchanges; Leung and
Nguyen~\cite{leung2019constructing} build Engle-Granger and Johansen cryptocurrency
portfolios and report tradable spread arbitrage. The closest prior to this paper is Fil and
Kristoufek~\cite{fil2020pairs}, who run distance and cointegration pairs on 26 Binance
majors and find that most strategies underperform their benchmarks once costs are realistic
and that the apparent intraday edge is highly cost-sensitive. Our honest-accounting
venue-adjusted Sharpe near $1.0$ on majors is consistent with their picture, and the
inflation we observe on the microcap held-out test matches the small-cap noise residual that
Wei, Brauneis-Mestel, and Sensoy each flag from a different angle.

\subsection{Machine learning approaches to pairs trading}
\label{sec:related-ml}

A line of work argues that deep learning extracts profitable structure from equity and
order-book data. Krauss, Do, and Huck~\cite{krauss2017deep} apply deep neural networks,
gradient-boosted trees, and random forests to S\&P~500 statistical arbitrage and find
ensemble lifts that erode sharply after costs and after the early-2000s regime. Fischer and
Krauss~\cite{fischer2018deep} extend the setting to LSTMs and report directional accuracy on
S\&P~500 constituents that decays once trading frictions are honest. Borovkova and
Tsiamas~\cite{borovkova2019ensemble} build an online LSTM ensemble for 5-minute equity bars,
where the high observation count gives deep models their best shot. On limit-order-book
data, Sirignano~\cite{sirignano2019deep} and Zhang, Zohren, and
Roberts~\cite{zhang2019deeplob} demonstrate that CNN and CNN-LSTM architectures recover real
predictive structure at sub-second horizons, and Sirignano and
Cont~\cite{sirignano2019universal} argue this structure is universal across stocks. Textbook
syntheses appear in Dixon, Halperin, and Bilokon~\cite{dixon2020mlfinance} and in the
methodological cautions of L\'opez de Prado~\cite{lopezdeprado2018advances}.

Two threads inform our negative ML result. First, the LOB-deep-learning papers predict at
horizons (sub-second to a few seconds) shorter than the hourly bars on which most pair
strategies operate, consistent with our finding that L2 order-flow information decays within
30 seconds and does not survive to a tradable horizon. Second, L\'opez de
Prado~\cite{lopezdeprado2018advances} and Harvey, Liu, and Zhu~\cite{harvey2016crosssection}
document that unadjusted multiple testing inflates apparent ML edges in finance; our own
ablation, in which a calibrated spread $z$-score carries roughly 79\% of the P\&L of a
four-architecture XGBoost/LSTM/transformer stack, sits squarely in that tradition.

\subsection{What is new here}
\label{sec:related-novelty}

Three things in this paper are not present in the literatures above. First, the
state-space and cointegration branches of pairs trading
(\cite{elliott2005pairs, triantafyllopoulos2011dynamic, vidyamurthy2004pairs,
caldeira2013selection, avellaneda2010statarb, fil2020pairs}) treat Kalman innovations and
rolling-$z$ deviations as tradable signals or as inputs to ADF or Johansen screens, but in
every case we surveyed the resulting screen is never benchmarked against a placebo that
cannot be cointegrated. Harvey's textbook fact~\cite{harvey1989structural} that a
well-specified filter has white steady-state innovations is well known in the state-space
community, but the implication that running ADF on those innovations renders the screen
circular is, to our knowledge, not made anywhere in pairs trading. The matched-placebo
audit that connects them is our headline methodological contribution. Second, we extend
Andersen and Bondarenko's~\cite{andersen2014vpin} critique of VPIN to crypto and quantify
the inversion: VPIN reads as a directional-consensus meter on Binance.com majors, not as
toxicity. Third, on the deployability side, we ship a structural-break circuit-breaker
analysis on point-in-time data that includes the LUNA, UST, FTT, and LUNC collapses, and
we show that a 50\%-single-leg halt preserves roughly 96\% of the monthly Sharpe at the
8\%/year breakage rate consistent with historical delistings. The survivorship and
circuit-breaker analysis on crypto pair spreads with explicit point-in-time-included dead
coins is also, to our knowledge, new.
```

---

## Section 3: Methods Appendix

Insert immediately before `\end{document}`. This is a verbatim drop-in. Note: the `\cite{}` keys inside have been standardized to match Section 1 (`saiddickey1984`, `contkukanovstoikov2014`, `hasbrouck1991measuring`, `easley2012vpin`, `johansen1991estimation`).

```latex
\appendix

\section{Methods background}
\label{sec:appendix}

This appendix collects working definitions of the tools used in the body of the paper. It is meant
for a reader who has seen probability and basic time series but may not have used all of these
specific instruments. We keep proofs out and give the formulas that are actually invoked in the
text. Section references in parentheses point to where the tool is used.

\subsection{Kalman filter and state-space form for dynamic hedge ratios}
\label{app:kalman}

Let $y_t$ be the dependent log price and $x_t$ the independent log price in a pair. The dynamic
hedge model treats the intercept $\alpha_t$ and slope $\beta_t$ as latent states that drift over
time. The state-space form is
\[
\theta_t \;=\; \theta_{t-1} + w_t, \qquad w_t \sim \mathcal{N}(0, Q),
\]
\[
y_t \;=\; H_t \, \theta_t + v_t, \qquad v_t \sim \mathcal{N}(0, R),
\]
with $\theta_t = (\alpha_t, \beta_t)^\top$, observation matrix $H_t = (1, x_t)$, state-noise
covariance $Q \in \mathbb{R}^{2\times 2}$, and observation-noise variance $R > 0$. The states
follow a random walk; only $y_t$ is observed.

Given the prior state mean $\hat{\theta}_{t-1|t-1}$ and covariance $P_{t-1|t-1}$, the Kalman
recursion predicts
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
The gain $K_t$ is the optimal blend of model prediction and new observation: it shrinks toward
zero when $R$ is large (data are noisy, trust the prior) and grows when $P_{t|t-1}$ is large
(prior is uncertain, take the data). The hyperparameters $Q$ and $R$ are estimated by maximizing
the Gaussian innovation likelihood
\[
\log \mathcal{L}(Q, R) \;=\; -\tfrac{1}{2} \sum_{t} \left( \log 2\pi S_t + e_t^2 / S_t \right),
\]
on a training window, then frozen and forward-rolled on the test window so no test-window
information leaks into the filter.

The mechanism behind the artifact reported in Section~\ref{sec:kalman} lives in the innovation
sequence. By construction, $\{e_t / \sqrt{S_t}\}$ has zero mean, unit variance, and zero serial
correlation under the model, and at steady state the filter approaches this whitening behaviour
for any input~\cite{harvey1989structural}. Running a stationarity test on $e_t$ therefore measures
how well the filter has been optimised, not whether $y_t$ and $x_t$ share a common stochastic
trend. A test that rejects the unit root on $e_t$ is consistent with no cointegration whatsoever.

\subsection{Augmented Dickey-Fuller test}
\label{app:adf}

The augmented Dickey-Fuller (ADF) test of Said and Dickey~\cite{saiddickey1984} tests the null
$H_0$: the series $z_t$ has a unit root (is $I(1)$) against $H_1$: $z_t$ is stationary ($I(0)$,
possibly around a constant or trend). The test regression is
\[
\Delta z_t \;=\; \mu + \rho \, z_{t-1} + \sum_{i=1}^{p} \gamma_i \, \Delta z_{t-i} + \varepsilon_t,
\]
with the augmentation lags $\Delta z_{t-i}$ included to soak up short-run serial correlation in
$\varepsilon_t$. The test statistic is the $t$-ratio on $\hat{\rho}$, compared against
Dickey-Fuller critical values (not standard normal). Under $H_0$, $\rho = 0$ and the process is
a random walk; under $H_1$, $\rho < 0$ and shocks decay geometrically. Reject for sufficiently
negative $t_{\hat\rho}$. We supplement ADF with the nonparametric Phillips-Perron
test~\cite{phillips1988testing} for robustness.

The connection to the rolling-$z$ artifact (Section~\ref{sec:kalman}) is mechanical: a series
that is close to white noise has trivially fast mean reversion, so the ADF rejects easily even
when no economic mean-reverting relation is present. The test answers ``is this stationary'',
not ``is this a cointegrating residual''.

\subsection{Engle-Granger cointegration}
\label{app:eg}

Two integrated series $y_t, x_t \sim I(1)$ are cointegrated if there exists a constant $\beta$
such that $y_t - \beta x_t$ is $I(0)$. The Engle-Granger two-step
procedure~\cite{engle1987cointegration} fits this $\beta$ by static OLS on levels,
\[
y_t \;=\; \alpha + \beta x_t + u_t,
\]
forms the residuals $\hat{u}_t = y_t - \hat\alpha - \hat\beta x_t$, and applies an ADF test to
$\hat{u}_t$ with Engle-Granger critical values (which are more conservative than ordinary ADF
values because $\hat\beta$ has been estimated). Rejection is evidence that the linear combination
defined by $(\hat\alpha, \hat\beta)$ is a stationary error-correction term. The multivariate
analogue is the maximum-likelihood VAR test of Johansen~\cite{johansen1991estimation}.

This is the clean comparison for the Kalman-innovations procedure of
Section~\ref{app:kalman}. Engle-Granger tests a residual whose dynamics are inherited from the
original prices; the Kalman procedure tests an innovation sequence whose dynamics are inherited
from the filter. The latter rejects the unit root on placebos that, by construction, cannot be
cointegrated; the former does not.

\subsection{Newey-West HAC standard errors}
\label{app:hac}

Ordinary OLS standard errors assume the regression errors are independent and homoskedastic. In
time series with overlapping returns, slow-moving regressors, or autocorrelated targets, both
assumptions fail and naive standard errors understate sampling uncertainty, often by a large
factor. The heteroskedasticity- and autocorrelation-consistent (HAC) covariance estimator of
Newey and West~\cite{newey1987hac} corrects this. For a regression with score $g_t$, the
Newey-West variance estimator is
\[
\hat{\Sigma}_{NW} \;=\; \hat{\Gamma}_0 + \sum_{\ell = 1}^{L} w_\ell \, \big( \hat{\Gamma}_\ell + \hat{\Gamma}_\ell^\top \big),
\qquad
w_\ell \;=\; 1 - \frac{\ell}{L+1},
\]
where $\hat{\Gamma}_\ell = T^{-1} \sum_{t = \ell+1}^{T} g_t g_{t-\ell}^\top$ is the sample
autocovariance at lag $\ell$ and $w_\ell$ is the Bartlett kernel. The kernel is non-negative and
linearly decreasing in $\ell$, which guarantees a positive semi-definite estimate.

Lag selection trades bias against variance; data-dependent rules are given by
Andrews~\cite{andrews1991hac}. We use $L = 60$ for one-hour analyses with a $60$-bar target,
$L = 24$ for daily analyses with a one-day target, and $L = 240$ for monthly Sharpe inference
where ten daily lags buffer the monthly aggregation. Sharpe ratios reported as ``HAC adjusted''
divide the mean by the square root of the HAC variance of the mean, then scale to monthly.

\subsection{Deflated Sharpe Ratio}
\label{app:dsr}

The deflated Sharpe ratio (DSR) of Bailey and L\'opez de Prado~\cite{bailey2014dsr} discounts a
reported Sharpe for non-normal returns and for the number of strategies that were tried before
the one that was reported. Let $\widehat{SR}$ be the in-sample Sharpe of the selected strategy,
$T$ the sample length in the same frequency as $\widehat{SR}$, $\gamma$ the sample skewness of
its returns, and $\kappa$ the sample kurtosis. The DSR is
\[
\mathrm{DSR} \;=\; \Phi\!\left( \frac{ (\widehat{SR} - SR_0) \, \sqrt{T - 1} }
    { \sqrt{1 - \gamma \, \widehat{SR} + \tfrac{\kappa - 1}{4} \, \widehat{SR}^{\,2} } } \right),
\]
where $\Phi$ is the standard normal CDF. The threshold $SR_0$ is the expected maximum Sharpe
under the null of zero true edge across $N$ independent trials,
\[
SR_0 \;=\; \sqrt{V[\,\widehat{SR}_n\,]} \left( (1 - \gamma_E) \, \Phi^{-1}\!\big(1 - 1/N\big)
    + \gamma_E \, \Phi^{-1}\!\big(1 - 1/(N e)\big) \right),
\]
with $\gamma_E \approx 0.5772$ the Euler-Mascheroni constant and $V[\widehat{SR}_n]$ the
cross-trial variance of estimated Sharpes under the null.

$N$ matters because the maximum of $N$ noisy estimates grows with $N$: trying ten strategies and
reporting the best one has a much higher null Sharpe than trying one. Doubling the number of
trials raises $SR_0$ by roughly $\sqrt{\log N}$ in the relevant regime, so a strategy that just
beat a one-trial null can fail a hundred-trial null. We use $N$ equal to the number of distinct
parameter or pair configurations evaluated before the reported one. For multi-model comparisons
we cross-check against Hansen's SPA test~\cite{hansen2005spa}.

\subsection{Ornstein-Uhlenbeck process and reversion speed}
\label{app:ou}

The Ornstein-Uhlenbeck (OU) process is the canonical continuous-time mean-reverting model. The
SDE is
\[
dS_t \;=\; \kappa \, (\mu - S_t) \, dt + \sigma \, dW_t,
\]
with long-run mean $\mu$, reversion speed $\kappa > 0$, instantaneous volatility $\sigma$, and
$W_t$ a Wiener process. The conditional mean decays geometrically toward $\mu$ with rate
$\kappa$, so the expected time for a deviation to halve is the half-life
\[
t_{1/2} \;=\; \frac{\ln 2}{\kappa}.
\]
We estimate $\kappa$ from a discretized AR(1) fit on the spread, $S_{t+1} - S_t = a + b (S_t - \mu) + \eta_t$
with $\hat{\kappa} = -\ln(1 + b) / \Delta t$, and rank pairs by this estimate.

The finding in Section~\ref{sec:real} that train-window $\hat\kappa$ predicts test-window
$\hat\kappa$ at Spearman $\rho = 0.46$ across walk-forward splits is a statement about the
process, not about profitability. A pair with a fast reversion speed today tends to have a fast
reversion speed next quarter, so $\kappa$ is a real, selectable property of pairs. It does not
follow that the deviations are tradable. Cost, capacity, and the speed at which $\mu$ itself
drifts can erase the edge.

\subsection{Walk-forward cross-validation}
\label{app:wf}

Walk-forward cross-validation is the standard non-anticipative analogue of $k$-fold CV for time
series. Fix a train length $T_{\mathrm{tr}}$, a test length $T_{\mathrm{te}}$, and a step length
$T_s$. Split $i$ uses observations $[t_i, t_i + T_{\mathrm{tr}})$ for fitting and
$[t_i + T_{\mathrm{tr}}, t_i + T_{\mathrm{tr}} + T_{\mathrm{te}})$ for evaluation, then advances
$t_{i+1} = t_i + T_s$. Any parameter that is fit (cointegrating $\beta$, OU $\kappa$, Kalman
$Q, R$, model weights) is refit per split using only the train window. Results are reported as
the distribution across splits, not as a single in-sample number.

The no-lookahead guarantee is that for every prediction at time $\tau$ in test split $i$, only
information from $[t_i, t_i + T_{\mathrm{tr}}) \subset [t_i, \tau)$ has been used. We use
$T_{\mathrm{tr}} = 6$ months, $T_{\mathrm{te}} = 3$ months, and $T_s = 3$ months in most of the
paper, with $90/30$ in earlier microstructure work. Random $k$-fold is inappropriate for time
series because it routinely places train observations after test observations and destroys this
guarantee.

\subsection{Placebo construction}
\label{app:placebo}

A placebo is a synthetic series with a known absence of the structure being tested. Different
nulls preserve different features, and the right placebo is the one that matches everything in
the data except the mechanism in question. We use four.

\emph{Random walks via cumulative sums.} Sample $\eta_t \sim \mathcal{N}(0, \sigma^2)$
i.i.d.~and set $z_t = z_{t-1} + \eta_t$. This preserves nothing about the original series except
the innovation variance; two independent random walks share no common trend. Anything that flags
common structure on this null is responding to filter mechanics or finite-sample artefacts.

\emph{Phase randomization.} Take the discrete Fourier transform $\tilde{z}_k = \mathcal{F}[z_t]$,
replace each non-DC phase by a uniform draw on $[0, 2\pi)$ subject to Hermitian symmetry, and
invert. The resulting series has the same power spectrum (autocovariance) as the original but
random phase, so all linear second-order statistics are preserved while any non-linear or
cross-series alignment is destroyed.

\emph{Block shuffle.} Partition $z_t$ into contiguous blocks of length $L$ and concatenate the
blocks in a random order. Short-range autocorrelation up to lag $\sim L$ is preserved; long-range
structure including unit roots and cointegrating drift is destroyed.

\emph{Random-pair pairing.} Keep each price series intact but pair it with the price series of
an unrelated coin. Marginal distributions, autocorrelations, volatility clustering, and trends
are all preserved leg by leg; only the joint cointegration is destroyed.

The four nulls answer different questions. The Kalman screen of Section~\ref{sec:kalman} fails
the random-walk null and the random-pair null at the same rate as the real data, which is the
diagnostic.

\subsection{Microstructure terms}
\label{app:micro}

\emph{Mid quote.} $p^{\mathrm{mid}}_t = (p^a_t + p^b_t)/2$, with $p^a_t$ the best ask and
$p^b_t$ the best bid.

\emph{Quoted spread.} $s_t = p^a_t - p^b_t$. The relative spread is $s_t / p^{\mathrm{mid}}_t$.

\emph{Microprice.} The size-weighted mid of Stoikov~\cite{stoikov2018microprice},
\[
p^{\mathrm{micro}}_t \;=\; \frac{q^b_t \, p^a_t + q^a_t \, p^b_t}{q^a_t + q^b_t},
\]
with $q^a_t, q^b_t$ the displayed sizes at the top of book. The microprice tilts toward the side
with less depth, since price is more likely to move that way next.

\emph{Order book imbalance.} For top $K$ levels,
\[
\mathrm{OBI}_t^{(K)} \;=\; \frac{\sum_{k=1}^{K} q^b_{t,k} - \sum_{k=1}^{K} q^a_{t,k}}
    {\sum_{k=1}^{K} q^b_{t,k} + \sum_{k=1}^{K} q^a_{t,k}} \;\in\; [-1, 1].
\]
Positive OBI means more displayed depth on the bid.

\emph{Order flow imbalance.} The best-level event-time OFI of Cont, Kukanov, and
Stoikov~\cite{contkukanovstoikov2014} is built from changes in displayed depth at the inside,
\[
e_t \;=\; \mathbb{1}\{p^b_t \geq p^b_{t-1}\} q^b_t - \mathbb{1}\{p^b_t \leq p^b_{t-1}\} q^b_{t-1}
    - \mathbb{1}\{p^a_t \leq p^a_{t-1}\} q^a_t + \mathbb{1}\{p^a_t \geq p^a_{t-1}\} q^a_{t-1},
\]
and the aggregated OFI over a window is
$\mathrm{OFI}_{[\tau_1, \tau_2]} = \sum_{t: \tau_1 \leq t < \tau_2} e_t$. The $e_t$ contribution
is positive for adds on the bid or lifts on the ask and negative for cancels on the bid or hits
on the bid, so OFI captures net inside-quote pressure.

\emph{VPIN.} The volume-synchronized probability of informed trading of Easley, L\'opez de
Prado, and O'Hara~\cite{easley2012vpin} groups trades into equal-volume buckets of size $V$ and
computes, within bucket $j$,
\[
\mathrm{VPIN}_j \;=\; \frac{1}{n} \sum_{i=j-n+1}^{j} \frac{|V_i^B - V_i^S|}{V},
\]
with $V_i^B$ and $V_i^S$ the buy- and sell-classified volume in bucket $i$ and $n$ the window
length in buckets. We classify volume by the Lee-Ready or bulk-volume rule; the construction
treats VPIN as a flow-imbalance meter rather than a true toxicity measure.

\emph{Permanent vs transient impact.} Hasbrouck~\cite{hasbrouck1991measuring} decomposes the
mid-quote response to a trade into a permanent component (information) that persists in the
efficient price and a transient component (liquidity) that decays as inventory works off. The
decomposition is the workhorse for separating informed flow from execution noise.

\subsection{Block bootstrap}
\label{app:bb}

The i.i.d.~bootstrap of Efron resamples observations independently with replacement, which is
invalid for autocorrelated series because it destroys the serial structure that drives sampling
uncertainty. The block bootstrap of K\"unsch~\cite{kunsch1989block} preserves it. From a series
$\{z_t\}_{t=1}^{T}$, draw $\lceil T / L \rceil$ blocks of length $L$ with replacement, where each
block is a contiguous slice $(z_s, z_{s+1}, \dots, z_{s+L-1})$ with $s$ uniform on
$\{1, \dots, T - L + 1\}$ (moving-block) or on a fixed grid (circular variants exist), concatenate
them, and truncate to length $T$. Resample $B$ times to form the bootstrap distribution. The
stationary bootstrap of Politis and Romano~\cite{politisromano1994stationary} randomizes the
block length and is preferable when the appropriate block size is uncertain.

The block length $L$ governs the bias-variance trade-off: $L$ must be large enough that adjacent
blocks are approximately independent, but small enough that enough distinct blocks exist. We use
$L = 300$ for hourly series, roughly twelve trading days, which is large relative to the
autocorrelation half-life of the spreads we examine and small relative to a full sample. Use the
block bootstrap whenever the statistic is sensitive to serial correlation, including Sharpe
ratios on returns, the mean of overlapping forecasts, and any quantity whose variance depends on
the spectrum of $z_t$ at low frequencies.
```

---

## Section 4: Figure Specifications

Numbered in priority order (must-have first, should-have, then nice-to-have). For each, the data source, the section to insert into, and a runnable sketch are given.

### Figure 1 (must-have) — `fig_kalman_innovations_white.pdf`

- **Caption:** A Kalman filter whitens its own innovations: representative real pair (top) and an independent random walk pair (bottom) produce visually indistinguishable innovation series, with sample ACFs flat past lag 1.
- **Section:** `sec:kalman` (Section 4.1) — visual companion to the placebo tables.
- **Data source:** Re-run `scratch/kalman_positive_control.py` or `scratch/audit_part1.py` to dump per-bar innovations for one real pair (e.g. BTCUSDT_ETHUSDT) and one independent-RW placebo. The same path produced `docs/CORRECTION_kalman_cointegration.md`.
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
- **Section:** `sec:kalman` (Section 4.1).
- **Data source:** `scratch/kalman_positive_control.csv` (columns: `kalman_adf_pass, kalman_innov_white, clean_eg_pass, static_adf_pass`, group `POS/NEG`).
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
- **Section:** `sec:real` (Section 5.1).
- **Data source:** `scratch/persistence_pairs.csv` (`split, pair, tr_kappa, oos_excess, oos_floor`).
- **Sketch:**
```python
df = pd.read_csv('scratch/persistence_pairs.csv')
fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
axes[0].scatter(df.tr_kappa, df.oos_excess, s=8, alpha=0.4)
rho = df[['tr_kappa','oos_excess']].corr('spearman').iloc[0,1]
axes[0].set_xlabel('In-sample OU kappa'); axes[0].set_ylabel('OOS excess reversion (z)')
axes[0].set_title(f'Real: Spearman rho = {rho:.2f}')
shuffled = df.copy()
shuffled['oos_excess'] = np.random.permutation(shuffled.oos_excess.values)
axes[1].scatter(shuffled.tr_kappa, shuffled.oos_excess, s=8, alpha=0.4, color='C3')
axes[1].set_title('Shuffled placebo: rho ~ 0')
plt.tight_layout(); plt.savefig('fig_reversion_persistence_scatter.pdf')
```

### Figure 4 (must-have) — `fig_circuit_breaker_retention.pdf`

- **Caption:** A structural-break circuit breaker preserves the no-stop reversion Sharpe across injected delisting rates; without it, the Sharpe collapses past 10%/year break rate.
- **Section:** `sec:survivor` (Section 6.4).
- **Data source:** `scratch/survivorship_adjusted_sharpe.csv` (`break_p_per_qtr, approx_annual, monthlyS_nobreaker, monthlyS_breaker, breaker_retain`).
- **Sketch:**
```python
df = pd.read_csv('scratch/survivorship_adjusted_sharpe.csv')
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(df.approx_annual*100, df.monthlyS_nobreaker, 'o-', label='No breaker', color='C3')
ax.plot(df.approx_annual*100, df.monthlyS_breaker, 's-',
        label='With circuit breaker (halt at 50% leg move)', color='C0')
ax.axhline(0, color='gray', lw=0.7, ls=':')
ax.set_xlabel('Injected delisting rate (% per year)')
ax.set_ylabel('Monthly Sharpe')
ax.legend(); ax.set_title('Circuit breaker absorbs survivorship tail')
plt.tight_layout(); plt.savefig('fig_circuit_breaker_retention.pdf')
```

### Figure 5 (must-have) — `fig_forced_collapse_perpair.pdf`

- **Caption:** Per-pair P\&L when the no-stop rule is forced to hold three real delisting collapses (LUNA, UST, FTT), with and without the structural-break circuit breaker.
- **Section:** `sec:survivor` (Section 6.4).
- **Data source:** `scratch/forced_collapse.csv` (`coin, collapse_date, nostop_pnl_pct, nostop_maxdd_pct, cb_pnl_pct, cb_maxdd_pct`).
- **Sketch:**
```python
df = pd.read_csv('scratch/forced_collapse.csv')
fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(df)); w = 0.35
ax.bar(x-w/2, df.nostop_pnl_pct, w, label='No stop, no breaker', color='C3')
ax.bar(x+w/2, df.cb_pnl_pct, w, label='With circuit breaker', color='C0')
ax.set_xticks(x); ax.set_xticklabels([f'{c}\n({d})' for c,d in zip(df.coin, df.collapse_date)])
ax.set_ylabel('Per-pair P&L (%)')
ax.axhline(-100, ls='--', color='gray', label='-100% (single-leg max)')
ax.legend(); plt.tight_layout(); plt.savefig('fig_forced_collapse_perpair.pdf')
```

### Figure 6 (should-have) — `fig_ofi_decay.pdf`

- **Caption:** Predictive incremental $R^2$ of book-OFI for forward midprice moves decays from $\sim$10--15\% contemporaneously to near zero by 30 seconds, across BTC/ETH/SOL/AVAX and all 2024 market regimes.
- **Section:** `sec:real` (Section 5.2).
- **Data source:** `scratch/book_ofi_2024.log` section `[2] PREDICTIVE incr book-OFI R^2 by horizon`. Parse or re-run `scratch/book_ofi_incremental.py` with `horizons = [1, 2, 5, 10, 30, 60, 120, 300]` seconds.
- **Sketch:**
```python
horizons = [1, 2, 5, 10, 30, 60, 120, 300]
fig, ax = plt.subplots(figsize=(7, 4))
for sym, color in zip(['BTCUSDT','ETHUSDT','SOLUSDT','AVAXUSDT'],
                      ['C0','C1','C2','C3']):
    ax.plot(horizons, r2_by_sym[sym], 'o-', label=sym, color=color)
ax.set_xscale('log'); ax.set_xlabel('Forward horizon (seconds)')
ax.set_ylabel('Incremental R^2 over trade-OFI')
ax.axhline(0, color='gray', lw=0.6)
ax.legend(); ax.set_title('Order-flow information half-life is seconds')
plt.tight_layout(); plt.savefig('fig_ofi_decay.pdf')
```

### Figure 7 (should-have) — `fig_microcap_adv.pdf`

- **Caption:** Median daily USD volume on Binance.US: the preregistered held-out microcaps (WAXP, ZIL, VTHO, ACH, EGLD) trade at hundreds to a few thousand dollars per day, three to four orders of magnitude below the majors whose costs the backtest assumes.
- **Section:** `sec:survivor` (Section 6.3).
- **Data source:** `scratch/microcap_adv.csv` (`symbol, median_daily_usd_vol`).
- **Sketch:**
```python
df = pd.read_csv('scratch/microcap_adv.csv').sort_values('median_daily_usd_vol')
fig, ax = plt.subplots(figsize=(7, 4))
microcaps = {'WAXPUSDT','ZILUSDT','VTHOUSDT','ACHUSDT','EGLDUSDT'}
colors = ['C3' if s in microcaps else 'C0' for s in df.symbol]
ax.barh(df.symbol, df.median_daily_usd_vol, color=colors)
ax.set_xscale('log')
ax.set_xlabel('Median daily USD volume (Binance.US, log scale)')
ax.set_title('Microcaps (red) vs majors (blue): 3-4 orders of magnitude apart')
plt.tight_layout(); plt.savefig('fig_microcap_adv.pdf')
```

### Figure 8 (should-have) — `fig_hac_inflation.pdf`

- **Caption:** Hourly Sharpe is heavily serial-correlation-inflated by multi-week holds: per-split IID vs Newey-West HAC ratios cluster around 1.5--2.5x across 27 walk-forward splits.
- **Section:** `sec:survivor` (Section 6.3).
- **Data source:** `docs/hac_sharpe_per_split.csv` (`inflation_factor, hac_pnl_mean_to_std_lag24, iid_pnl_mean_to_std, model`).
- **Sketch:**
```python
df = pd.read_csv('docs/hac_sharpe_per_split.csv')
df = df[df.model == 'zscore_rule']
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
axes[0].hist(df.inflation_factor.dropna(), bins=15, color='C0', edgecolor='k')
m = df.inflation_factor.median()
axes[0].axvline(m, color='C3', ls='--', label=f'median={m:.2f}x')
axes[0].set_xlabel('IID / HAC std ratio'); axes[0].set_ylabel('Splits'); axes[0].legend()
axes[1].scatter(df.iid_pnl_mean_to_std, df.hac_pnl_mean_to_std_lag24, s=20)
lims = [-0.5, 0.7]
axes[1].plot(lims, lims, 'k--', lw=0.7)
axes[1].plot(lims, [v*0.5 for v in lims], 'gray', ls=':', label='HAC = 0.5 x IID')
axes[1].set_xlabel('IID Sharpe'); axes[1].set_ylabel('Newey-West HAC Sharpe')
axes[1].legend()
plt.tight_layout(); plt.savefig('fig_hac_inflation.pdf')
```

### Figure 9 (nice-to-have) — `fig_cancellation_share.pdf`

- **Caption:** Cancellations dominate trades in best-level liquidity withdrawal: 81--85\% of best-level size reductions are cancels, invisible to trade-only microstructure analysis (BTC/ETH/SOL/AVAX, 2024).
- **Section:** `sec:real` (Section 5.2).
- **Data source:** `scratch/book_ofi_2024.log` cancel-share percentages, or re-run `scratch/book_ofi_cancel_stretch.py`.
- **Sketch:**
```python
syms = ['BTCUSDT','ETHUSDT','SOLUSDT','AVAXUSDT']
cancel_pct = [83, 81, 85, 82]  # parsed from log
trade_pct = [100-c for c in cancel_pct]
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.barh(syms, cancel_pct, color='C0', label='Cancellations')
ax.barh(syms, trade_pct, left=cancel_pct, color='C1', label='Trades')
ax.set_xlabel('Share of best-level liquidity withdrawal (%)')
ax.set_xlim(0, 100); ax.legend(loc='lower right')
plt.tight_layout(); plt.savefig('fig_cancellation_share.pdf')
```

### Existing figure: reshape `fig_freq_invariance.pdf`

Do NOT cut it. Mihai's complaint is accurate as-drawn (it duplicates Table 3 percentages with bars), but the underlying point — that real and placebo pass at identical rates across hourly/4h/daily — is the visual punchline of the Kalman artifact section. Reshape to show the **placebo gap** rather than the levels:

- For each frequency (hourly, 4h, daily), plot `(real - placebo)` Kalman pass-rate gap at $p<0.05$ and $p<0.001$, alongside `(real - placebo)` for the clean Engle-Granger test.
- The Kalman screen's gap is essentially zero at every frequency (the artifact); the clean-test gap is large and frequency-dependent. A flat zero line for the Kalman delta vs. a tall sloping bar for the clean delta lands the message Table 3 cannot.

Keep `fig_rollingz_noise.pdf` as is — it shows the mechanism on one path and is genuinely additive to Table 4.

---

## Section 5: New Table Specifications

### Table T1 (must-have) — `tab:venue-comparison`

- **Caption:** Naive vs HAC-adjusted monthly Sharpe on Binance.US and Coinbase for the no-stop reversion strategy, showing the venue gap ($\sim$2.2 vs $\sim$0.88) and the serial-correlation adjustment (Binance 2.18 $\to$ 1.4).
- **Section:** `sec:survivor` (Section 6.3).
- **Data source:** `docs/hac_sharpe_fixed.csv` and `docs/hac_sharpe_fixed_deep.csv`.
- **Why:** The paper buries the four-number Sharpe story (naive Binance, HAC Binance, naive Coinbase, HAC Coinbase) in prose. A small table makes the "centered $\sim$1.0" claim auditable in one glance and is the single most important number in Section 6.

### Table T2 (should-have) — `tab:exec-symbol-breakdown`

- **Caption:** Execution costs by symbol and order size show aggressive crossing dominates for all four majors at both \$10k and \$50k notionals; the L3-from-L2 signal never beats the random-rate placebo.
- **Section:** `sec:real` (Section 5.2).
- **Data source:** `scratch/exec_value_2024_summary.csv` (`sym, size, agg, naive30, l3_30, l3_posted_pct`).
- **Why:** Current Table 4 pools across symbols and sizes, hiding that the null holds uniformly. Per-symbol breakdown serves Mihai's "self-contained for non-experts" ask and makes the BTC vs AVAX cost gap visible.

### Table T3 (should-have) — `tab:institutional-impact`

- **Caption:** Per-trade price impact at 1-second horizon by trade size bucket and symbol: institutional ($>$\$10k) consistently 2.0--2.6x retail across BTC/ETH/SOL/AVAX, validating the size-as-information proxy.
- **Section:** `sec:real` (Section 5.2).
- **Data source:** `scratch/impact_decomp_2024.log` (parse seconds-horizon impact-by-size-bucket).
- **Why:** Per-symbol numbers (BTC 0.33/0.23/0.13, ETH 0.32/0.24/0.15, SOL 0.32/0.26/0.15, AVAX 0.38/0.38/0.18) currently appear only in `L2_FINDINGS`, not in the paper. A small table sharpens the only positive microstructure finding.

---

## Section 6: Contributions Sharpening (Replacement Intro Paragraph)

Replace the current "We make three contributions" paragraph (lines 56--68 of `main.tex`) with the following ~250-word block. It is sharper about what is new vs. known, in line with Mihai's request and the novelty note from the Pairs lit review.

```latex
We make three contributions, and we are explicit about which are methodological and which are
empirical. The first is methodological and is, to our knowledge, new. Two natural looking
cointegration screens manufacture the appearance of structure on data that cannot be
cointegrated. A Kalman dynamic hedge ratio reports out of sample cointegration in almost every
liquid pair because it runs a stationarity test on a filter's own innovations, which are
white by construction. A rolling $z$ score reports strong mean reversion because it reverts
mechanically even on a pure random walk. The state-space pairs trading literature
(\cite{elliott2005pairs, triantafyllopoulos2011dynamic}) and the wider cointegration screening
tradition (\cite{vidyamurthy2004pairs, avellaneda2010statarb, caldeira2013selection,
leung2019constructing, fil2020pairs}) build the machinery we audit, and the textbook fact that
a well specified filter has white steady state innovations (\cite{harvey1989structural}) is
classical, but the implication that this renders the cointegration screen circular, and a
matched placebo audit that exposes it on independent random walks, phase-randomized series,
and block-shuffled series at hourly, four-hour, and daily frequencies, are not in the prior
literature. The second contribution is empirical: the structure that survives is not tradable.
Order flow in the Level 2 book carries real contemporaneous information that decays within
30 seconds and cannot fund the execution costs we measure on 18,432 simulated orders; reversion
speed is a selectable property of pairs (Spearman $\rho = 0.46$ out of sample) but the ranking
does not convert into profit. The third is deployability: a structural-break circuit breaker
on point-in-time data including LUNA, UST, FTT, and LUNC preserves roughly 96\% of the
monthly Sharpe of a no-stop reversion book at a realistic 8\%-per-year delisting rate.
```

---

## Section 7: Priority Order

Apply changes in this order. Each item's dependencies are flagged.

1. **Bibliography merge (Section 1).** Add all new `\bibitem` entries first, keeping the four already in the paper. Verify no duplicate keys. **Blocker for:** every later section, since they all cite new keys.
2. **Standardize the four ambiguous keys** across all new content: `contkukanovstoikov2014`, `easley2012vpin`, `hasbrouck1991measuring`, `saiddickey1984`, `johansen1991estimation`. Search-and-replace any earlier draft text that used the alternate keys.
3. **Replace the contributions paragraph (Section 6)** in `\section{Introduction}`. Self-contained, only depends on bibliography for `\cite` resolution. Quick win — do this before Related Work so the intro frames the new related-work scope.
4. **Replace `\section{Related work}` (Section 2)** with the four-subsection version. ~1500 words. Depends on bibliography (step 1). Read the prose end-to-end after pasting to check the transitions between subsections.
5. **Append Methods Appendix (Section 3)** before `\end{document}`. Depends on bibliography (step 1). Verify `\appendix` is not already declared earlier in the document — if so, just paste the `\section{Methods background}` and following `\subsection` blocks without the `\appendix` directive.
6. **Add Table T1 `tab:venue-comparison` (Section 5).** Must-have. Plain LaTeX, no figure rendering required. Insert in Section 6.3.
7. **Generate must-have figures (Section 4, Figures 1--5).** These rely on data already in the repo:
   - Fig 1, 2: need a rerun of `scratch/kalman_positive_control.py` if innovation series aren't already cached.
   - Fig 3: `scratch/persistence_pairs.csv` ready.
   - Fig 4: `scratch/survivorship_adjusted_sharpe.csv` ready.
   - Fig 5: `scratch/forced_collapse.csv` ready.
   No bibliography dependencies, but the captions use terminology defined in the appendix (Kalman innovations, OU $\kappa$, etc.), so do step 5 first if you want the figure captions to reference appendix sections.
8. **Reshape `fig_freq_invariance.pdf`** per the note in Section 4. This is a separate task from the new figures and should be done in the same scripting session.
9. **Generate should-have figures and tables (Figures 6--8, Tables T2--T3).** Fig 6 may require parsing `scratch/book_ofi_2024.log` or rerunning `scratch/book_ofi_incremental.py` with the right horizon grid. Fig 8 needs `docs/hac_sharpe_per_split.csv` filtered to the `zscore_rule` model. Tables T2, T3 are data the agent has but the parsing of `impact_decomp_2024.log` for T3 may be the slowest step.
10. **Generate Figure 9 (nice-to-have) `fig_cancellation_share.pdf`** if time permits.
11. **Title.** The paper title is still `TBD`. Suggested working title aligned with the contributions: *"Screening artifacts in cryptocurrency pair spreads: why Kalman cointegration tests pass random walks, and what survives once they don't"*. Decide before submission.
12. **Final pass.** Once all sections are in, recompile and: (a) check every `\cite{}` resolves; (b) check `\label{}` / `\ref{}` for `sec:kalman`, `sec:real`, `sec:survivor` haven't drifted; (c) check the new figure file names match what's in `\includegraphics{}`; (d) word count and figure/table count for the three-student-project length target.

**Cross-dependency summary.** The bibliography is the only hard prerequisite for everything else. Within the prose changes, the contributions paragraph can be pasted independently. Within the figure work, Figures 1--5 (must-have) are independent of Figures 6--9 (should/nice). The reshaped `fig_freq_invariance.pdf` is its own item. Apply in the numbered order above for the fewest re-compiles.

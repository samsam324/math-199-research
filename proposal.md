For this research project and paper, I’m looking at whether the price of various cryptocurrencies tend to move in a stable, quantifiable relationship with one another over time, and whether these relationships deviate in a way that corrects over time. A good analogy for this is two complementary goods that tend to have a stable relationship with one another in terms of price, such that if one of the edges of a market gets overpriced relative to the other, this relationship corrects over time. In quantitative finance, this concept of using statistics to identify and test such relationships that correct and stabilize over time is called statistical arbitrage.

While I don’t assume that markets are irrational or that such a strategy would necessarily be profitable, I do hope to quantify the degree to which such relationships exist in cryptocurrency markets, and the stability with which they do so, as a way of gaining a deeper understanding of market efficiency and information flow within a large and frequently traded market.

I will retrieve data from a public cryptocurrency exchange API named Binance. The data will include historical hourly data for spot trading pairs with USDT, a crypto coin that nearly 1 to 1 follows the US dollar, commonly used as a unit of account balance in cryptocurrency markets. Each data point will include the open, high, low, and close prices for each hour, as well as volume traded in each asset pair for each hour. The data will be locally stored in a structured format to allow for reproducibility of analysis without relying on redownloading the data.

This process has two phases. The first phase, research and preparation, will create a clean data set and clarify what is actually tradable at a given moment. The data will be cleaned to create a coherent data set with all dates in UTC, no duplicates, monotone timestamps, all assets on the same hourly schedule, and documentation on any missing data or irregularities. The rules for handling missing hours must also be clear to avoid the creation of false patterns in the data. The investible universe, the set of coins actually available to be traded at a given moment, is also important to understand. The cryptocurrency markets are constantly changing with new coins being added and others having little history, so when evaluating any historical period, the investible universe must be determined using only information available at the time to avoid the use of future knowledge when making future predictions.

In the analysis phase, I will look for pairs of coins that seem to have a consistent relationship. A liquid coin is one that tends to trade frequently with large volume, such that it can be bought or sold with minimal impact on the price. Liquid coins are desirable in the market because trading them is cheap, safe, and less susceptible to distortion or manipulation from individual traders. I will focus on liquid coins because I want to look at pairs that are likely to be tradable. I will look at a pair of coins and define a spread, which is a normalized difference between the two price series. A spread is a good concept to use in this analysis because it turns two sets of moving prices into a single series that measures how far the pair of assets has deviated from a normal relationship. I will then check if this tends to revert towards a normal level after deviating. This will be a good indication of mean reversion in the relative prices. I will also check if a simple trading rule based on this spread performs well in a historical simulation. I will use this approach to make it easier to quantify if this effect is strong enough to be useful after accounting for realistic limitations. I will use a walk forward approach to evaluate this trading strategy, which means I will select pairs based on past data and then check them on future data. This is a good approach to ensure that I do not inadvertently tune the approach to get a particular result based on future data that I would not have had access to at the time.

This project is inspired by established academic work on pairs trading and relative value strategies, particularly Gatev, Goetzmann, and Rouwenhorst 2006, which studies a systematic pairs trading rule and provides a methodological baseline for selecting and evaluating pairs. I will adapt this general idea to the cryptocurrency setting, where markets operate continuously and the set of tradable assets changes quicker over time.

Reference:
Gatev, E., Goetzmann, W. N., and Rouwenhorst, K. G. (2006). Pairs Trading: Performance of a Relative Value Arbitrage Rule. Review of Financial Studies, 19(3), 797–827.
Please provide a short description of the research project you will be working on with faculty supervision. Include a brief description of how you will proceed both in the research phase and in your approach to the final project.
Please indicate the tangible evidence you will provide as proof of the work completed. Evidence of work may be in the form of a culminating paper or project.

---

# Remaining Work — Task Breakdown

## 1. Dataset Completion

### 1a. Liquidity filtering
- [ ] Compute average hourly volume (in USDT terms) per symbol over the training window
- [ ] Pick a threshold (e.g., top 50-80 symbols by volume, or a minimum USDT volume floor)
- [ ] Add this as a filter step between universe selection and pair scoring
- [ ] The volume data is already in the parquet files (the `volume` column) — no new data needed

### 1b. Survivorship bias documentation
- [ ] For each walk-forward window, log which symbols drop out mid-window (data stops before window end)
- [ ] Count how many symbols were delisted between training and test periods
- [ ] Write a short section for the paper acknowledging the limitation: we only have currently-listed coins, delisted coins are missing, bias direction overstates strategy performance

## 2. Feature Engineering (New Columns Derived from Existing Data)

These all come from data you already have — no new downloads needed.

### 2a. Spread features (per pair, per timestep)
- [ ] Spread value: OLS residual from log-price regression (already computed in pair_selection.py)
- [ ] Spread z-score: (spread - rolling mean) / rolling std, using a lookback window (e.g., 168h = 1 week)
- [ ] Spread rate of change: first difference of spread over last 1h, 4h, 24h
- [ ] Spread volatility: rolling std of spread (e.g., 24h and 168h windows)

### 2b. Volume features (per pair, per timestep)
- [ ] Volume ratio: volume_A / volume_B (relative activity between the two legs)
- [ ] Volume z-score: each symbol's volume relative to its own rolling mean
- [ ] Combined pair volume: sum or geometric mean of both legs' USDT volume

### 2c. Regime/context features
- [ ] BTC return over trailing 24h (market-wide trend proxy — when BTC dumps, correlations spike and spreads blow out)
- [ ] Rolling correlation between the two symbols (does the relationship tighten or loosen over time?)
- [ ] Realized volatility of each leg (rolling std of log returns, e.g., 24h window)

## 3. Classical Analysis (Your Collaborator's Scope)

### 3a. Hidden Markov Model on spreads

**Goal:** Detect when a pair is in a mean-reverting regime vs. a trending/diverging regime, and use that as a trade filter.

**Library:** `hmmlearn` (Gaussian HMM is standard for continuous spread data)

**Concrete steps:**
- [ ] Compute the spread series for each top-scoring pair (already done by `pair_selection.py` — extract residuals)
- [ ] Engineer HMM input features per timestep: spread z-score, spread first difference, rolling spread volatility (24h)
- [ ] Fit a 2-state Gaussian HMM per pair (state 1 = mean-reverting, state 2 = diverging). Start with 2 states; can extend to 3 if results warrant.
- [ ] Decode the state sequence with Viterbi — assigns each timestep to the most likely state
- [ ] Identify which state is "mean-reverting" by checking which has lower spread volatility and faster autocorrelation decay
- [ ] Use as a trade filter: only enter trades when the model says we're in the mean-reverting state
- [ ] Walk-forward fit: refit the HMM at each test boundary using only past data
- [ ] Compare backtest results with and without the HMM filter (Sharpe, win rate, max drawdown)

**Expected output:** A plot per pair showing spread + colored backgrounds for HMM-decoded states, plus a comparison table of backtest metrics with/without filter.

### 3b. Kalman filter for dynamic hedge ratios

**Goal:** Replace static OLS beta with a beta that adapts over time as the relationship between two coins evolves.

**Library:** `pykalman` or write directly with numpy (state-space form is standard)

**Concrete steps:**
- [ ] State-space model: y_t = beta_t * x_t + alpha_t + noise, where beta_t and alpha_t evolve as random walks
- [ ] Initialize Kalman filter with the static OLS estimates as the prior mean
- [ ] Tune observation noise and process noise (the latter controls how fast beta is allowed to drift) via maximum likelihood on training data
- [ ] Run the filter forward over the spread to get a time-varying beta_t
- [ ] Recompute spreads using dynamic beta: spread_t = y_t - beta_t * x_t - alpha_t
- [ ] Compare static vs. dynamic spreads on:
  - ADF p-value (does the dynamic spread look more stationary?)
  - Backtest Sharpe (does the trade rule perform better?)
  - Visual: overlay static spread vs. dynamic spread on the same chart

**Expected output:** Plot of beta_t over time per pair, comparison of static vs. dynamic spread quality, backtest metric table.

## 4. ML/DL Models (Your Scope)

All three models use the same input features (section 2), the same labels, and the same walk-forward splits — so results are directly comparable.

### 4a. Shared data preparation

**Goal:** Produce a single dataset that all three models train on, so comparisons are clean.

**Concrete steps:**
- [ ] For each cointegrated pair (top ~20-50 from section 1), extract the spread series from the existing `pair_selection.py` output
- [ ] Build sliding windows of length W = 168 hours (1 week), stride = 1 hour
- [ ] Per-timestep features (from section 2): spread z-score, spread first difference, spread volatility (24h), volume ratio, BTC 24h return, rolling correlation (24h), realized volatility of each leg
- [ ] Label options (pick one, do both if time permits):
  - **Regression**: spread change over next N=24 hours (continuous target)
  - **Classification**: 3-class label — revert (spread moves toward 0), persist (stays in same band), diverge (moves further from 0)
- [ ] Time-based splits: rolling walk-forward windows, e.g., 90 days train → 30 days test, advance by 30 days
- [ ] Save the prepared dataset as parquet so all models load from the same source

**Output:** A reusable dataset object/file that takes a (pair, time window) and returns (X, y).

### 4b. Model 1: XGBoost baseline (start here)

**Library:** `xgboost`

**Why first:** Handles tabular features natively, no sequence modeling needed, fast to train, gives feature importance for free, most likely to perform well on this data size.

**Concrete steps:**
- [ ] Flatten each window into a single tabular row of summary features: latest spread z-score, mean/min/max spread z-score over window, spread slope (linear fit over window), latest volume ratio, mean volume ratio, BTC 24h return, latest rolling correlation, realized volatility of each leg, time-since-last-zero-crossing
- [ ] Train XGBoost regressor (`XGBRegressor`) for the regression target, or classifier (`XGBClassifier`) for the 3-class target
- [ ] Walk-forward training: retrain on all data prior to each test boundary
- [ ] Hyperparameter search (small grid): max_depth ∈ {3, 4, 5, 6}, n_estimators ∈ {100, 300, 500}, learning_rate ∈ {0.01, 0.05, 0.1}
- [ ] Use early stopping with a 10% held-out slice from the end of the training data
- [ ] Extract `feature_importances_` and SHAP values — both directly answer "what drives spread mean reversion?"

**Output:** Predicted signal for each (pair, timestep) in test windows + feature importance rankings.

### 4c. Model 2: LSTM (deep learning)

**Library:** PyTorch (`torch.nn.LSTM`)

**Why:** Natural fit for sequential data with limited samples. Fewer parameters than a transformer, well-suited for our window-length-168 inputs.

**Concrete steps:**
- [ ] Architecture: 1-2 layer LSTM, hidden dim 64, dropout 0.2 between layers, final linear head
- [ ] Input shape: (batch, 168, num_features) — full sequence, not flattened
- [ ] Output: scalar (regression) or 3-class softmax
- [ ] Loss: MSE (regression) or cross-entropy (classification)
- [ ] Optimizer: Adam, lr=1e-3, weight_decay=1e-5
- [ ] Batch size: 256-512
- [ ] Standardize features per training window (z-score using only training data statistics)
- [ ] Early stopping on validation slice (last 10% of training period) with patience=10
- [ ] Walk-forward retrain at each test boundary (or fine-tune from prior weights to save compute)
- [ ] Track training/validation loss curves for the paper

**Output:** Predicted signal per (pair, timestep) in test windows.

### 4d. Model 3: Transformer (comparison)

**Library:** PyTorch (`torch.nn.TransformerEncoder`)

**Why:** Tests whether attention adds value over recurrence. Keep small to avoid overfitting.

**Concrete steps:**
- [ ] Architecture: 2-layer transformer encoder, 2 attention heads, d_model=32, feedforward dim=64, dropout=0.2
- [ ] Sinusoidal positional encoding over the 168-step window
- [ ] [CLS]-style pooling: prepend a learnable token, use its final embedding for prediction (or use mean pooling)
- [ ] Linear head: 1 output for regression, 3 for classification
- [ ] Same loss/optimizer/training regime as LSTM
- [ ] Capture attention weights from the final layer for visualization
- [ ] For 3-4 example trades, plot attention as a heatmap (which historical hours does the model focus on?) — this is the figure that distinguishes the transformer in the paper

**Output:** Predicted signals + attention weight visualizations.

### 4e. Evaluation (head-to-head comparison)

**Goal:** Apples-to-apples comparison of all three models on the same data, fed through the same backtester.

**Concrete steps:**
- [ ] Predictive metrics on test windows: directional accuracy, MSE/MAE (regression) or precision/recall/F1 (classification), AUC if classification
- [ ] Trading metrics (feed signals to minitron): Sharpe ratio, total return, max drawdown, win rate, average trade P&L
- [ ] Statistical significance: bootstrap or paired t-test on per-trade returns to test if differences between models are real or noise
- [ ] Ablation: drop one input feature at a time, measure degradation for each model
- [ ] Sanity check baselines: compare against (a) the existing classical z-score threshold rule, (b) random predictions
- [ ] Final figure: bar chart of Sharpe ratio across all models + classical baseline, with confidence intervals

**Paper narrative options:**
- "Tree-based ML matches or beats DL on this problem" — valid, interesting finding for limited-data quant
- "DL beats classical, transformer beats LSTM" — supports investing in DL for cross-asset signals
- "Nothing beats the classical baseline" — also a valid finding, important for the literature

## 5. Cross-Market Lead-Lag Analysis (Prediction Markets vs. Spot)

This is a separate but complementary research angle: do prediction markets price in information about crypto before spot markets do?

### 5a. Data collection
- [ ] Pull historical contract prices from Kalshi API for crypto binary contracts ("BTC above $X by date Y")
- [ ] Focus on contracts with multiple strike levels sharing the same expiry (e.g., BTC above $95k, $100k, $105k all for same date)
- [ ] Record mid-prices at regular intervals (hourly to match Binance data)
- [ ] Store locally in the same parquet format as Binance data for consistency
- [ ] Document available strike spacing (typically $5k for BTC), date ranges, and liquidity

### 5b. Implied price derivation
Each contract "BTC above K" at price p gives P(BTC > K) = p. A set of contracts at different strikes traces out the survival function directly.

- [ ] For each timestamp and expiry, collect prices p(K_1), p(K_2), ..., p(K_n) across all strikes
- [ ] Enforce monotonicity (p should decrease as K increases — thin markets may violate this)
- [ ] Compute implied CDF: CDF(K) = 1 - p(K)
- [ ] Compute implied expected price: E[S] = Σ p(K_i) · ΔK (integral of survival function)
- [ ] Optional: fit a lognormal or mixture distribution to smooth the blocky implied PDF from coarse strike spacing
- [ ] Flag and exclude stale quotes (far OTM/ITM strikes with no recent trades)

Reference: this is the prediction market analog of Breeden-Litzenberger (1978), which extracts risk-neutral densities from option prices.

### 5c. Lead-lag analysis
- [ ] Compute the spread: implied expected price minus Binance spot price
- [ ] Test Granger causality in both directions — does the prediction market lead spot, or vice versa?
- [ ] Cross-correlation at various lags (1h, 2h, 4h, 12h, 24h) to find the dominant lead-lag structure
- [ ] Control for the fact that implied prices are risk-neutral (embed risk premia), not pure forecasts

### 5d. Informed trade detection (isolation forest)
- [ ] Pull trade-level or order book data from Kalshi API for crypto contracts
- [ ] Engineer features per time window: volume spike (relative to rolling average), price impact (price move per unit volume), bid-ask spread compression, order imbalance
- [ ] Fit isolation forest on these features to flag anomalous windows
- [ ] Test whether flagged anomalies in prediction markets predict subsequent spot price moves on Binance
- [ ] Measure: what is the average spot return in the N hours following a detected anomaly vs. baseline?

### 5e. Paper narrative
- [ ] Academic framing: lead-lag between derivatives and spot is well-studied in options (Stephan & Whaley 1990, Easley et al. 1998 show informed trading hits derivatives first). The prediction market version is largely unstudied — this is a genuine research gap.
- [ ] Note that prediction market prices are risk-neutral expectations, not forecasts — discuss implications
- [ ] If prediction markets lead spot: evidence that prediction markets aggregate information faster than spot exchanges
- [ ] If spot leads prediction markets: evidence that prediction markets are slow to reprice, creating exploitable inefficiency

## 6. What You Need Right Now (Priority Order)

1. **Build the feature engineering pipeline** (section 2) — prerequisite for the ML models, uses existing Binance data
2. **Add liquidity filtering** (section 1a) — narrows the universe to realistic pairs
3. **Pull Kalshi historical data** (section 5a) — new data needed for the lead-lag analysis, can run in parallel with steps 1-2
4. **Prepare shared training data** (section 4a) — depends on section 2 being done
5. **Build XGBoost baseline first** (section 4b) — fastest to implement, sets the bar
6. **Build LSTM** (section 4c) — your main DL model
7. **Build transformer** (section 4d) — comparison model, do last
8. **Implied price derivation + lead-lag analysis** (sections 5b-5c) — can run in parallel with ML work
9. **Informed trade detection** (section 5d) — depends on Kalshi data
10. **Head-to-head evaluation** (section 4e) — the results section of the paper

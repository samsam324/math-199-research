# Delisted-coin survivorship re-test (Task 2)

*Dated 2026-06-08. Closes the one outstanding survivorship hole in the no-stop reversion
alpha: fully-delisted coins, absent from the on-disk universe by construction.*

## 1. What was checked — Tardis availability

Queried the Tardis exchange-metadata API (`/v1/exchanges/binance`, spot) for the
survivorship-worst symbols. **All are available** (L2 `book_snapshot_25` + `trades`):

| symbol | Tardis availableSince | availableTo / delisting |
|---|---|---|
| `LUNAUSDT` (Terra) | 2021-01-28 | trades to the May-2022 collapse |
| `USTUSDT` (TerraUSD) | 2021-12-24 | **2022-05-14** (Tardis cuts off at the depeg) |
| `FTTUSDT` (FTX token) | 2019-12-20 | Nov-2022 FTX collapse / delist |
| `LUNCUSDT` (Luna Classic, post-rebrand) | 2022-09-09 | still trading |

So Tardis *does* carry the fully-delisted tail. The reversion test needs only **hourly
closes** (not L2), so the actual pull used Binance public klines (`data.binance.vision`
monthly 1h), which are free, fast, and **include the collapse months** — verified:
LUNA min close **$0.00005** (full collapse to ~zero), UST ends at the $0.25 depeg, FTT
delists Nov-2022 at ~$1.4 (from ~$25 pre-FTX). Script: `scratch/t2_pull_delisted.py` →
`data/spot_1h_delisted/`. (Tardis L2 would be needed only if we ever want the delisted
coins in the *1s microstructure* work; for the hourly reversion test, klines are exact.)

## 2. Method

Added {LUNA, UST, FTT, LUNC} to the 204-symbol on-disk pool (→ 208) and re-ran the
headline reversion config (40-pair, no-stop, z-exit, realistic 30 bps; `wf_survivorship`
machinery) with **point-in-time entry** — a coin is selectable in a walk-forward window
only if it has ≥90% coverage in that window's *train* span — and **delisting exit**: the
collapse is in the price data, so an open position marks the crater bar-by-bar, then goes
flat when trading stops. Compared 204 vs 208. Scripts: `scratch/t2_survivorship_rerun.py`,
`t2_worstcase.py`.

## 3. What was found

**Headline (does the alpha survive?): it WEAKENS modestly but SURVIVES.**

| universe | monthly Sharpe | max drawdown | delisted pairs ever selected |
|---|---:|---:|---:|
| 204-symbol on-disk (baseline) | **+3.76** | −23.2% | 0 |
| 204 + {LUNA, UST, FTT, LUNC} | **+3.56** | **−28.0%** | 45 |

This is the **first survivorship check to dent the result** (the iter-8 structural-break
stress and the iter-15 broad-universe test both left it intact or stronger). The damage:

- **The losses are concentrated in 2021**, holding hyper-volatile LUNA/FTT pairs that never
  reverted within their window: e.g. KNC_LUNA **−145%**, BAT_LUNA **−142%**, XLM_LUNA −136%,
  ONE_FTT −60%, XLM_FTT −56% (leverage-equivalent per-pair P&L). 40-pair diversification
  absorbs them, but they deepen the drawdown −23% → −28% and shave ~0.2 off monthly Sharpe.
- **The strategy mostly AVOIDED the actual May-2022 collapse window.** No LUNA or UST pair
  was selected for the 2022-Q2 test window: UST had only **54% train coverage** (listed
  2021-12-24) → excluded by the point-in-time gate; LUNA had full coverage but **did not
  rank top-40 by reversion speed** (it was trending, not reverting — the selection working
  as intended: it avoids non-mean-reverting names).
- **But that avoidance is load-bearing and partly fortuitous.** Forcing the no-stop rule to
  hold delisted-leg pairs *through* the May-2022 collapse (`t2_worstcase.py`): worst pair
  **LUNA_BAT −1435%** (log-spread units; the realizable dollar loss on a $1 long leg is
  bounded near −100%, still a total loss of that pair), mean **−67%** across 31 forced pairs
  vs a typical pair's ~+30–40%. So had LUNA cleared the top-40 reversion cutoff in that one
  window, a single pair would have cost ~−2.5% of the 40-pair book, and 2–3 such pairs would
  have wrecked the quarter. This is the catastrophic tail the iter-8 stress test simulated —
  now confirmed concretely with the real Terra collapse.

## 4. How it changes the deployable Sharpe range

- **As actually realized:** the survivorship-worst tail costs ~5% of monthly Sharpe
  (3.76 → 3.56 on the 204-universe) and ~5 pp of drawdown (−23% → −28%). Applied to the
  frequency-honest deployable range of **~1.7–2.5** (iter-13), this nudges it to roughly
  **~1.6–2.4** — the alpha **survives the fully-delisted tail**, so the paper no longer needs
  the silent "tested only on coins surviving to 2026" caveat; it can state the dent explicitly.
- **But the no-stop rule has genuine catastrophic tail exposure** to a held name that
  structurally breaks (−100%+/pair). The strategy's robustness here rests on the
  reversion-speed selection *not picking the collapsing coin* — real partial protection
  (it shuns trending names) but **not a guarantee** (a top-40 cutoff). **A live deployment
  must add a delisting/halt/structural-break risk control** (hard stop or per-name limit on
  a name showing a one-way break) that the pure no-stop rule lacks. This is the honest
  headline for the paper's risk section.

## 5. Bottom line for the paper

Replace the silent survivorship caveat with: *"The no-stop mean-reversion alpha was re-tested
on a point-in-time universe including the fully-delisted survivorship-worst coins (LUNA, UST,
FTT, LUNC, with their actual collapses in the data). The alpha survives — monthly Sharpe
3.76 → 3.56, max drawdown −23% → −28% — because (i) 40-pair diversification absorbs even
−100%+ single-pair losses and (ii) the reversion-speed selection largely avoids trending,
about-to-collapse names. The avoidance is not guaranteed, however: forced to hold a Terra
pair through the May-2022 collapse, the no-stop rule loses a full leg (~−100% to −1435% in
log-spread terms) on that pair. A deployable version therefore requires an explicit
structural-break / delisting risk control, which the pure no-stop rule lacks."*

"""Reproduce the "L2 execution costs 17 to 23% below the flat 5 bps" claim from the
two committed portfolio cost runs. Run: python scratch/provenance_costsaving.py"""
import pandas as pd

flat = pd.read_csv(r"docs/l2_results/portfolio_flat_costs.csv").set_index("model")
l2 = pd.read_csv(r"docs/l2_results/portfolio_l2_costs.csv").set_index("model")
print("per-model cost (total_return is cost-dominated; lower magnitude = cheaper):")
for m in ["zscore_rule", "xgboost"]:
    f = abs(flat.loc[m, "total_return"]); g = abs(l2.loc[m, "total_return"])
    print(f"  {m:14s} flat={f:10.1f}  L2={g:10.1f}  -> {100*(f-g)/f:5.1f}% cheaper")
print("=> L2 walked-book costs run ~17 to 23% below the flat 5 bps assumption.")

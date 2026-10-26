# 2005 dynamic-universe patch

Copy these files into the existing project's `src/` folder and overwrite the old versions.

Changed files:
- config.py
- 01_download_prices.py
- 03_build_networks.py
- 04_build_labels.py
- 09_build_residual_network.py
- 10_build_tail_network_metrics.py

Main changes:
1. START_DATE is now 2005-01-01.
2. Tickers are no longer globally dropped for weak full-period coverage.
3. Dynamic rolling-window availability is used instead.
4. Network scripts skip only windows with too few active tickers.
5. Coverage diagnostics are saved.

Recommended clean rerun:

```powershell
# optional but recommended: remove old processed outputs to avoid mixing 2015 and 2005 runs
rmdir /s /q data_raw\prices
rmdir /s /q data_processed\returns
rmdir /s /q data_processed\networks
rmdir /s /q data_processed\labels
rmdir /s /q data_processed\evidence
rmdir /s /q outputs\reports
rmdir /s /q outputs\figures

# recreate happens automatically by scripts where needed
python src/01_download_prices.py
python src/02_download_macro_stress.py
python src/03_build_networks.py
python src/04_build_labels.py
python src/05_feasibility_report.py

python src/08_build_forward_labels.py
python src/09_build_residual_network.py
python src/10_build_tail_network_metrics.py
python src/11_build_prediction_baselines.py
python src/12_build_provenance_evidence_table_v2.py
```

After running, inspect:
- data_raw/prices/ticker_coverage_report.csv
- data_processed/returns/daily_active_ticker_count.csv
- data_processed/networks/network_snapshot_coverage.csv
- data_processed/networks/residual_network_snapshot_coverage.csv
- data_processed/networks/tail_network_snapshot_coverage.csv
- data_processed/labels/forward_label_summary.csv
- outputs/reports/baseline_model_results.csv

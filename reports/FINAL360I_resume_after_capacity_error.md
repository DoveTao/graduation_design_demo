# FINAL360I resume after capacity error

- capacity error summary: `Selected model is at capacity` occurred after FINAL360I training and evaluation had already completed, during the commit/push wrap-up stage.
- retraining executed during resume: `false`
- evaluation-only recovery executed during resume: `false`
- artifacts verified: `true`
- metrics source: `reports/FINAL360I_metrics_val.json`, `reports/FINAL360I_metrics_test.json`, `reports/FINAL360I_model_selection_table.json`
- verified main report: `reports/FINAL360I_final_retrain_and_model_selection.md`
- verified selected checkpoint present locally: `/home/dovetao/graduation_design_demo/checkpoints/FINAL360I_struct360b_final/seed0/best_val.pt`
- verified final checkpoint present locally: `/home/dovetao/graduation_design_demo/checkpoints/FINAL360I_struct360b_final/seed0/final.pt`
- checkpoint committed to git: `false`
- first recovery commit hash: `5933e2c`
- first recovery push status: `success`
- branch: `experiment/final360i-final-retrain-and-model-selection`
- note: `Checkpoint artifacts remain preserved locally and were intentionally not committed.`

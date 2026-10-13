# Does a sepsis early-warning model fail women?

Machine learning summative project. We predict sepsis in ICU patients up to 6 hours before it is clinically recognised, then check whether the models warn women as early and as often as men.

* **Problem:** binary classification for every ICU hour. Will this patient meet the Sepsis-3 criteria within the next 6 hours?
* **Data:** PhysioNet / Computing in Cardiology Challenge 2019. 40,336 ICU patients from two US hospital systems, 1,552,210 hourly rows, 40 clinical variables. [Dataset page](https://physionet.org/content/challenge-2019/1.0.0/)
* **Models:** Logistic Regression, Random Forest and Histogram Gradient Boosting (scikit-learn), plus a feed-forward network (Keras Sequential) and a GRU over each patient's hourly record (Keras Functional). All Keras inputs go through `tf.data`.
* **Fairness checks:** recall, false alarms, share of septic patients warned, and hours of warning, all compared for women and men. We also test removing sex from the inputs, reweighting, and per-sex thresholds, with bootstrap confidence intervals.

## Links

* Report: _add link_
* Google Drive (raw data zip and best models): _add link_
* Video: _add link_

## Repository layout

```
sepsis_fairness.ipynb   main notebook, runs end to end
results/experiments.csv experiment table (also logged to TensorBoard)
figures/                plots used in the report
logs/                   TensorBoard logs: hparams/ (every experiment), fit/ (Keras training curves), graph/ (GRU graph)
requirements.txt
```

The raw data and the trained models are not in this repository. They are in the Google Drive folder.

## How to run

1. Download the data, either from the Drive zip or from PhysioNet:
   ```bash
   wget -r -N -c -np https://physionet.org/files/challenge-2019/1.0.0/
   ```
   Put the folders so that you have `data/raw/training_setA/*.psv` and `data/raw/training_setB/*.psv`.
2. Install the packages (Python 3.10 to 3.12):
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Run the notebook (training everything takes about 17 minutes on a laptop CPU (Apple M-series, 16 GB RAM); trained models are cached in `models/runs/` and `data/processed/runs/`, so later runs take about 2 minutes. Set `RETRAIN=1` to train again):
   ```bash
   jupyter nbconvert --to notebook --execute sepsis_fairness.ipynb --inplace
   ```
   Set `QUICK=1` first for a fast check on 3,000 patients with 1 epoch.
4. Open TensorBoard:
   ```bash
   tensorboard --logdir logs
   ```

## Results

All numbers are on the held-out test patients (15%, never used for training or model choice). Models were chosen on validation PR-AUC. Alerts use an alarm budget: the threshold is set on validation so that 5% of non-sepsis hours raise an alert. Recall is per hour. Warning time counts the first alert in the 24 hours before onset. E12 and E13 are trained on hospital A and tested on hospital B.

| Run | Model | ROC-AUC | PR-AUC | Recall F / M | False alerts per day F / M | Warning median F / M (h) |
|---|---|---|---|---|---|---|
| E1_logreg | LogisticRegression | 0.778 | 0.084 | 0.28 / 0.29 | 0.49 / 0.58 | 20.0 / 19.0 |
| E2_random_forest | RandomForest | 0.834 | 0.098 | 0.42 / 0.31 | 0.46 / 0.28 | 14.0 / 23.5 |
| E3_hgb_default | HistGradientBoosting | 0.831 | 0.097 | 0.34 / 0.32 | 0.55 / 0.50 | 14.5 / 17.0 |
| E4_hgb_tuned | HistGradientBoosting | 0.807 | 0.082 | 0.29 / 0.28 | 0.65 / 0.61 | 14.0 / 16.0 |
| E5_hgb_no_sex | HistGradientBoosting | 0.804 | 0.081 | 0.27 / 0.28 | 0.65 / 0.55 | 12.0 / 14.0 |
| E14_hgb_no_time | HistGradientBoosting | 0.820 | 0.095 | 0.32 / 0.31 | 0.44 / 0.57 | 15.0 / 17.0 |
| E6_mlp | Keras MLP (Sequential) | 0.829 | 0.115 | 0.32 / 0.30 | 0.47 / 0.53 | 13.5 / 18.0 |
| E7_gru64 | Keras GRU (Functional) | 0.819 | 0.089 | 0.31 / 0.29 | 0.73 / 0.65 | 11.0 / 17.0 |
| E8_gru128x2 | Keras GRU (Functional) | 0.827 | 0.110 | 0.35 / 0.31 | 0.57 / 0.51 | 13.0 / 16.0 |
| E9_gru64_noweight | Keras GRU (Functional) | 0.820 | 0.096 | 0.30 / 0.30 | 0.66 / 0.71 | 11.0 / 15.0 |
| E10_hgb_reweight_sex | HistGradientBoosting | 0.806 | 0.079 | 0.27 / 0.28 | 0.59 / 0.60 | 13.0 / 13.0 |
| E11_sex_thresholds | E8_gru128x2 + per-sex threshold | 0.827 | 0.110 | 0.35 / 0.31 | 0.57 / 0.52 | 13.0 / 16.0 |
| E12_hgb_A_to_B | HistGradientBoosting | 0.672 | 0.040 | 0.12 / 0.11 | 0.20 / 0.28 | 17.0 / 19.0 |
| E13_gru_A_to_B | Keras GRU (Functional) | 0.743 | 0.054 | 0.12 / 0.16 | 0.25 / 0.49 | 10.0 / 6.0 |

Main observations:

* Most models land around ROC-AUC 0.82 to 0.83. The two-layer GRU (E8) had the best validation PR-AUC (0.110), followed by the MLP; gradient boosting and random forest were close behind and train in seconds.
* Recall is similar for women and men. The 95% bootstrap interval of the recall gap includes zero for every model except the random forest, which catches more sepsis hours in women but also raises more false alerts for them (`results/bootstrap_recall_gap.csv`).
* In 10 of 12 models women get a shorter median warning than men (about 2 to 6 hours), but almost all intervals include zero, so this is a trend that needs more data, not a proven gap (`results/bootstrap_warning_time.csv`). All models are tested on the same 178 septic women, so the 10 of 12 are not independent.
* Removing sex from the inputs (E5) or reweighting women (E10) changed little. For E8, per-sex thresholds (E11) came out almost equal to the shared threshold, because recall was already equal on the validation set.
* Removing the ICU-time features (E14) lowered ROC-AUC only from 0.831 to 0.820, so the models do not depend on that shortcut alone.
* Moving to a new hospital hurts a lot: gradient boosting drops to ROC-AUC 0.67 at hospital B and the GRU to 0.74.
* Lab tests are taken less often for women in this data (lactate about 17% and pH about 20% less often per hour), one possible reason for any gap.

## Data licence and citation

The PhysioNet Challenge 2019 data is open access (the project page lists CC BY 4.0, and the LICENSE file in the release is ODbL). Please cite:

* M. A. Reyna et al., "Early Prediction of Sepsis From Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019," *Critical Care Medicine*, vol. 48, no. 2, pp. 210-217, 2020.
* A. L. Goldberger et al., "PhysioBank, PhysioToolkit, and PhysioNet," *Circulation*, vol. 101, no. 23, pp. e215-e220, 2000.

# Thermostat Override Prediction — Binary Classification

Predicting when a smart thermostat's programmed schedule will be overridden by the homeowner, using one year of five-minute interval data from a single US household Ecobee thermostat. This project compares a simple Logistic Regression baseline against a Random Forest "kitchen sink" model, with full EDA, feature engineering, and production deployment analysis.

---

## Overview

Smart thermostats learn and adapt to user behavior over time. This project treats manual override events as a proxy for how well the thermostat understands homeowner preferences. Accurately predicting overrides enables proactive schedule adjustment, reducing manual intervention and improving comfort without user effort.

---

## Dataset

| Property | Detail |
|---|---|
| Source | Ecobee smart thermostat (single US household) |
| Period | January 1 – December 31, 2017 |
| Frequency | 5-minute intervals |
| Total records | 105,120 |
| Target | Override (Binary: Override / No Override) |
| Override rate | 16.75% (17,603 override events) |

**Original features:** Event type, schedule type, indoor temperature, indoor humidity, outdoor temperature, outdoor humidity, cool setpoint, heat setpoint

**Train/test split:** 80/20 chronological split (Jan–Oct 2017 train, Oct–Dec 2017 test) — preserves temporal ordering to prevent data leakage from future observations into training.

---

## Key EDA Findings

- **Hour of day:** Override rate ranges from 9% (3 AM) to 31% (8 PM) — strong daily cycle
- **Day of week:** Friday peaks at 25%; Saturday drops to 12%
- **Seasonality:** January hits 28% override rate; July drops to 8%
- **Smart Away:** 0% override rate when occupancy detection triggers — perfect separator
- **Schedule type:** Home schedule (21% override) vs Sleep (13%)
- **Heat setpoint:** Strongest numeric predictor (r = 0.57 with target)

---

## Feature Engineering

Starting from 8 raw features, 35 additional features were engineered for a total of 43:

**Temporal:** Hour, day of week, month, day of year, week of year, weekend indicator, morning/evening peak flags, time period and season categories

**Comfort metrics:** Temperature relative to cool/heat setpoints, deadband width, in-deadband indicator, distance from setpoint midpoint, too-hot/too-cold flags

**Weather dynamics:** Indoor/outdoor temperature differential, 1-hour temperature change, rapid change indicator, extreme weather flags (computed from training quantiles to prevent leakage)

**Behavioral:** Schedule change indicator, setpoint change indicator, hour × weekend interaction, temperature × season interaction, rolling moving averages

---

## Models

### Model 1 — Logistic Regression (Simple Baseline)
- 6 carefully selected features after VIF analysis
- Indoor_CoolSetpoint removed due to extreme multicollinearity (VIF = 342) with HeatSetpoint
- 5-fold cross-validation to tune regularization strength (C)
- Class weights balanced to handle 17% override rate

### Model 2 — Random Forest (Kitchen Sink)
- All 48 encoded features included — no manual selection
- GridSearchCV with 3-fold CV across 48 parameter combinations
- Naturally handles multicollinearity and non-linear relationships
- Class weights balanced

---

## Results

| Metric | Model 1 (Logistic Regression) | Model 2 (Random Forest) |
|---|---|---|
| Features | 6 | 48 |
| CV ROC-AUC | 0.8541 | 0.9796 |
| Train ROC-AUC | 0.8522 | 1.0000 |
| **Test ROC-AUC** | **0.6945** | **0.9984** |
| Test Accuracy | 91.39% | 99.32% |
| Test Precision | 98.44% | 95.87% |
| Test Recall | 34.51% | **99.09%** |
| Test F1-Score | 0.5111 | 0.9745 |

**Model 2 is recommended for production.** The Logistic Regression missed 1,795 of 2,741 override events (65% false negative rate). The Random Forest missed only 25 (0.9%).

---

## Precision-Recall Trade-off

In a smart thermostat context, **false negatives are more costly than false positives**:

- **False negative:** System fails to predict an override → user must manually adjust → defeats the purpose of a "learning" thermostat
- **False positive:** System preemptively adjusts → minor energy cost, but user perceives responsiveness

**Recommendation:** Lower the classification threshold from 0.5 to ~0.35 to push recall toward 99.5%+, accepting a modest increase in false positive rate (from 0.64% to ~1–2%). Personalize thresholds per user over time based on observed tolerance for unnecessary adjustments.

---

## Top Features (Random Forest)

| Feature | Importance |
|---|---|
| deadband_width (CoolSetpoint − HeatSetpoint) | 17.8% |
| Indoor_HeatSetpoint | 14.5% |
| Indoor_CoolSetpoint | 13.0% |
| setpoint_midpoint | 12.0% |
| temp_below_heat | 5.2% |

Engineered comfort features — not raw temperatures — dominate. Domain knowledge applied to feature engineering outperformed raw sensor readings.

---

## How Smart is the Ecobee? (Single Household Assessment)

**Rating: 6/10** — based on this one household's 2017 data

| What earns points | What loses points |
|---|---|
| Smart Away: 0% override when away (+2) | 16.75% annual override rate — ~48/day (-2) |
| Basic schedule execution (+2) | No temporal pattern adaptation (-1) |
| Some schedule-type awareness (+1) | Relies on user-configured setpoints, not learned preferences (-1) |
| User preference compliance (+1) | |

Our Model 2, using the same sensor data the thermostat has access to, achieves 99.09% recall — demonstrating the hardware and data are sufficient for much smarter behavior. The limitation is in current algorithms, not available information.

---

## Tech Stack

- Python, scikit-learn, pandas, NumPy, Matplotlib, Seaborn
- `LogisticRegression`, `RandomForestClassifier`
- `GridSearchCV`, `cross_val_score`
- `roc_auc_score`, `roc_curve`, `confusion_matrix`, `classification_report`
- `variance_inflation_factor` (statsmodels)

---

## Repository Structure

```
thermostat-override-prediction/
├── notebook.ipynb    # Full pipeline: EDA, feature engineering, modeling, evaluation
├── README.md
```

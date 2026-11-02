# Fraud Detection

Detecting fraudulent card transactions when fraud is rare, using a time based evaluation that mirrors how a model would be used in practice.

## Data

The [Kaggle credit card fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud): 284,807 transactions and 492 frauds (0.17%). Features are 28 anonymized PCA components plus Time and Amount. The data is not included, so download `creditcard.csv` to run this.

```bash
pip install -r requirements.txt
python analysis.py path/to/creditcard.csv
```

## Method

- **Split by time:** train on the earliest 60%, choose the decision threshold on the next 20%, and test once on the latest 20%. The test period contains 75 frauds.
- **No scaling leakage:** scalers sit inside pipelines, so they are fit on training data only.
- **Metrics:** PR-AUC is the main metric, because accuracy is uninformative at this fraud rate. PR-AUC ranges are 95% bootstrap intervals.
- **Threshold:** chosen to maximize F1 on the validation period. A real system would use the cost of missed fraud and false alarms instead.

## Results on the test period

| Model | PR-AUC | PR-AUC range | ROC-AUC | Precision | Recall | Fraud dollars caught |
| --- | --- | --- | --- | --- | --- | --- |
| Majority class baseline | 0.001 | 0.001 to 0.002 | 0.500 | 0.001 | 1.000 | n/a |
| Logistic regression | 0.693 | 0.579 to 0.788 | 0.973 | 0.699 | 0.680 | 48.2% |
| Logistic regression, class weights | 0.744 | 0.636 to 0.851 | 0.982 | 0.943 | 0.667 | 56.8% |
| Random forest | 0.791 | 0.699 to 0.869 | 0.947 | 0.902 | 0.733 | 65.3% |
| Gradient boosting, default settings | 0.310 | 0.216 to 0.405 | 0.740 | 0.471 | 0.653 | 39.6% |

Full output is in `results.json`.

## Reading the results

- Random forest has the best PR-AUC, but its range overlaps weighted logistic regression. With only 75 test frauds, this does not show that it is clearly better, and the simpler model may be preferable where decisions need explaining.
- ROC-AUC and PR-AUC rank the models differently, which is why ROC-AUC alone is not enough on rare events.
- Default gradient boosting performed poorly. It was not tuned, so this says nothing about boosting in general.

## Limitations

- No hyperparameter tuning, and no probability calibration beyond a Brier score check.
- One time split, so results depend on this period.
- Anonymized features prevent business interpretation of what drives a prediction.
- No drift monitoring or fairness analysis.

## Project history

`coursework/fraud_detection_course_notebook.ipynb` is the original Johns Hopkins Applied Machine Learning notebook, with its outputs unchanged. It used a partial file of 25,837 rows, a random split, and a scaler fit before the split, and the PyTorch models never flagged fraud in their short training runs.

`analysis.py` is a later rerun on the full dataset that fixes these issues, built with AI coding assistance.

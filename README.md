# Diabetes Risk Prediction

The Diabetes Risk Prediction Dataset is a large-scale synthetic healthcare dataset created for machine learning, data science, healthcare analytics, and predictive modeling. It contains 50,000 patient records with 41 clinical, lifestyle, demographic, and health-related features designed to simulate real-world diabetes risk assessment scenarios.

Dataset: [Kaggle - Diabetes Risk Prediction Dataset (50K patients)](https://www.kaggle.com/datasets/mobeenfatimah/diabetes-risk-prediction-dataset-50k-patients)

This project builds a machine learning model that predicts diabetes risk from those features.

## Approach

1. **Cleaning:** drop duplicates and rows with a missing target; drop ID-like columns.
2. **Leakage check:** flag any single feature that predicts the target almost perfectly (AUC >= 0.97).
3. **Preprocessing:** median imputation and scaling for numeric features; mode imputation and one-hot encoding for categorical features, all inside a `Pipeline` so nothing leaks from the test set.
4. **Model comparison:** Logistic Regression, Random Forest and Histogram Gradient Boosting, compared with 5-fold stratified CV on ROC-AUC.
5. **Tuning:** randomized search on the best model.
6. **Evaluation:** held-out 20% test set with accuracy, macro-F1, ROC-AUC, PR-AUC, a confusion matrix, an ROC curve and permutation feature importance.

## Quick start

```bash
pip install -r requirements.txt

# Download the data from Kaggle (or download manually and place the CSV in data/)
kaggle datasets download -d mobeenfatimah/diabetes-risk-prediction-dataset-50k-patients -p data --unzip

python train.py --data data/<your_file>.csv --target <target_column>
python predict.py --input new_patients.csv --output predictions.csv
```

If `--target` is omitted, the script tries to guess it from the column names. It supports binary targets and multi-class risk levels (Low / Medium / High).


## Results

_Run `train.py`.

## Limitations

- The data is **synthetic**. Scores reflect how the data was generated, not real-world clinical performance.
- This is an educational project and **not a medical device**. It must not be used for diagnosis or treatment decisions.
- If the target is derived from columns like glucose or HbA1c, accuracy will look unrealistically high. Check the leakage warning in the training log.

## Project structure

```
├── data/           
├── models/          
├── reports/         
├── train.py
├── predict.py
└── requirements.txt
```

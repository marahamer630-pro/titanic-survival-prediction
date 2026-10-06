# Titanic Survival Prediction: KNN vs Decision Tree

A classification project that predicts whether a Titanic passenger survived, and compares two models: K-Nearest Neighbors and a Decision Tree. University project for the Machine Learning Lab course (Arab International University).

## What this project does

- Loads the Titanic dataset (**891 passengers**, 12 columns)
- Cleans and prepares the data: drops `Name` and `Ticket`, encodes `Sex`, groups `Cabin` by deck letter (missing values become "Unknown"), fills missing `Embarked` values, and imputes missing `Age` with the mean
- Splits the data 80/20 (stratified) and scales features with `StandardScaler`
- Tests KNN with K from 1 to 20 and picks the best K (K = 14)
- Trains a Decision Tree with the entropy criterion (information gain, as in ID3) and `max_depth=4`, and plots the tree
- Evaluates both models with confusion matrices, classification reports and accuracy scores

## Results (179-passenger test set)

| Model | Accuracy |
|-------|----------|
| KNN (K = 14) | **82.68%** |
| Decision Tree (entropy, depth 4) | 78.77% |

KNN confusion matrix: `[[103, 7], [24, 45]]`. KNN is strong at identifying passengers who did not survive (recall 0.94) and weaker on survivors (recall 0.65).

## Tech stack

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, Jupyter Notebook

## Run it

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook titanic_survival_prediction.ipynb
```

Keep `Titanic-Dataset.csv` in the same folder as the notebook.

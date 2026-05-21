# Spaceship Titanic

This repository contains code for the Kaggle competition:  
[Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic)

## Project Overview

The goal of this competition is to predict whether a passenger was transported to another dimension during the Spaceship Titanic accident.

This project implements a complete machine learning pipeline including:

- Data preprocessing
- Feature engineering
- Missing value handling
- Hyperparameter tuning
- Cross-validation
- Final model training
- Submission generation

The model used in this project is **CatBoostClassifier**, which naturally handles categorical features and performs well on tabular datasets.

---

## Data Processing

### Feature Engineering

Several additional features are created from the original dataset.

#### Cabin decomposition

The `Cabin` column is split into:

- `Deck`
- `CabinNum`
- `Side`

Example:

```text
B/123/P
↓
Deck = B
CabinNum = 123
Side = P
```

---

#### Passenger group information

Extract group information from `PassengerId`:

- `Group`
- `MemberID`
- `GroupSize`
- `IsAlone`

Example:

```text
0001_01
↓
Group = 0001
MemberID = 01
```

Additional features:

- `GroupSize`
- `IsAlone`

---

#### Spending features

Total spending:

```python
TotalSpend = RoomService + FoodCourt + ShoppingMall + Spa + VRDeck
```

Additional indicators:

- `NoSpending`
- `Cryo_NoSpend`

---

#### Age grouping

Passengers are divided into age groups:

```text
0–12
12–18
18–25
25–40
40–60
60–100
```

---

#### Log transformation

The following features use logarithmic transformation:

```python
np.log1p()
```

Applied to:

- RoomService
- FoodCourt
- ShoppingMall
- Spa
- VRDeck
- TotalSpend

This reduces skewness caused by extreme values.

---

### Missing Value Handling

#### Categorical features

Missing values are filled with:

```python
"Unknown"
```

Columns:

- HomePlanet
- Destination
- VIP
- CryoSleep
- Deck
- Side

#### Numerical features

Missing values are replaced by the median value from the training set.

---

## Model Training

### Cross Validation

Stratified K-fold validation:

```python
n_splits = 5
```

Advantages:

- Preserves class distribution
- Reduces overfitting risk
- Produces more reliable estimates

---

### Hyperparameter Search

Grid search over:

```python
depth: [5, 10, 15]

learning_rate:
[0.05, 0.1]

l2_leaf_reg:
[1, 3]
```

Each parameter combination is evaluated using:

```python
Mean CV Accuracy
```

---

### Final Model

The best parameters found during cross-validation are used for final training:

```python
final_model = CatBoostClassifier(
    iterations=5000,
    depth=best_depth,
    learning_rate=best_lr,
    l2_leaf_reg=best_l2,
    random_seed=42,
    early_stopping_rounds=300
)
```

---

## Prediction and Submission

Generate predictions:

```python
test_preds = final_model.predict(X_test)
```

Create submission file:

```python
submission.csv
```

Format:

| PassengerId | Transported |
|-------------|-------------|
| 0013_01     | True        |
| 0018_01     | False       |

---

## Method Summary

Pipeline:

```text
Raw Data
    ↓
Feature Engineering
    ↓
Missing Value Processing
    ↓
Cross Validation
    ↓
Hyperparameter Search
    ↓
CatBoost Training
    ↓
Prediction
    ↓
submission.csv
```

---

## License

This project is for educational and research purposes.

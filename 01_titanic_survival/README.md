# Titanic Survival Prediction

Goal: predict whether a passenger survived from simple passenger information.

This is a beginner classification project. The label is `survived`, which is either `0` or `1`.

## Run

```bash
python 01_titanic_survival/train.py
```

## Things To Interrogate

- What is a feature?
- What is a label?
- Why do we split into train and test data?
- What does `.fit()` do?
- Why do we need different preprocessing for numbers and categories?
- What does accuracy measure?

## Modifications

- Remove `fare` from the feature list.
- Add `age_group` manually from `age`.
- Change `LogisticRegression` to `RandomForestClassifier`.
- Change `test_size` from `0.25` to `0.4`.
- Add a new passenger to the prediction examples.

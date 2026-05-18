# House Price Regression

Goal: predict house price from size, bedrooms, bathrooms, age, and distance to the city.

This is a beginner regression project. The label is `price_k`, meaning price in thousands.

## Run

```bash
python 02_house_price_regression/train.py
```

## Things To Interrogate

- What is regression?
- What does `mean_absolute_error` mean?
- Why do we compare a baseline model against a trained model?
- What does `LinearRegression` learn?
- What does a coefficient mean?

## Modifications

- Add a new feature like `has_garage`.
- Remove `distance_to_city_km`.
- Try `RandomForestRegressor`.
- Predict three houses of your own.
- Sort the errors and inspect the worst prediction.

from pathlib import Path
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


#build location access for the file
DATA_PATH = Path(__file__).parent / "data" / "houses_tiny.csv"


def main():
# load the data into panda (spreadsheet in python)
    data = pd.read_csv(DATA_PATH)

# list of inputs 
    feature_columns = [
        "size_sqft",
        "bedrooms",
        "bathrooms",
        "age_years",
        "distance_to_city_km",
    ]
# list of outputs
    target_column = "price_k"

#separateing the x as input and y as output
    X = data[feature_columns]
    y = data[target_column]

# split the training data in to testing and training
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=7,
    )

# average pricing for all housing data
    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_train, y_train)
    baseline_predictions = baseline.predict(X_test)

# create an linear regression model, then train and predict prices for houses unseen
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)



    print("House price model")
    print(f"Rows: {len(data)}")
    print()
    print("Baseline mean absolute error:")
    print(f"${mean_absolute_error(y_test, baseline_predictions):.1f}k")
    print()
    print("Linear regression metrics:")
    print(f"Mean absolute error: ${mean_absolute_error(y_test, predictions):.1f}k")
    print(f"Mean squared error: {mean_squared_error(y_test, predictions):.1f}")
    print(f"R^2 score: {r2_score(y_test, predictions):.2f}")
    print()

    coefficients = pd.Series(model.coef_, index=feature_columns)
    print("Learned coefficients")
    print(coefficients.sort_values(ascending=False))
    print(f"Intercept: {model.intercept_:.2f}")
    print()

#predict new house
    example_houses = pd.DataFrame(
        [
            {
                "size_sqft": 1600,
                "bedrooms": 3,
                "bathrooms": 2,
                "age_years": 10,
                "distance_to_city_km": 6.0,
            },
            {
                "size_sqft": 2600,
                "bedrooms": 5,
                "bathrooms": 4,
                "age_years": 4,
                "distance_to_city_km": 3.0,
            },
        ]
    )

    example_predictions = model.predict(example_houses)

    print("Example predictions")
    for house, price in zip(example_houses.to_dict("records"), example_predictions):
        print(f"{house} -> ${price:.0f}k")


if __name__ == "__main__":
    main()

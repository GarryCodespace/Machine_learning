from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path(__file__).parent / "data" / "titanic_tiny.csv"

def main():
    data = pd.read_csv(DATA_PATH)

    feature_columns = [
        "pclass",
        "sex",
        "age",
        "sibsp",
        "parch",
        "fare",
        "embarked",
    ]

    target_column = "survived"

# select only these columns 
    X = data[feature_columns]
    y = data[target_column]

    numeric_features = [
        "pclass",
        "age",
        "sibsp",
        "parch",
        "fare",
    ]

    categorical_features = [
        "sex",
        "embarked",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("numbers", StandardScaler(), numeric_features),
            ("categories", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
)
    
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(classification_report(y_test, predictions))

    print(accuracy)


    example_passengers = pd.DataFrame(
        [
            {
                "pclass": 3,
                "sex": "male",
                "age": 24,
                "sibsp": 0,
                "parch": 0,
                "fare": 8.0,
                "embarked": "S",
            },
            {
                "pclass": 1,
                "sex": "female",
                "age": 31,
                "sibsp": 1,
                "parch": 0,
                "fare": 82.0,
                "embarked": "C",
            },
        ]
    )

    example_predictions = model.predict(example_passengers)
    example_probabilities = model.predict_proba(example_passengers)[:, 1]

    print("Example predictions")
    for passenger, prediction, probability in zip(
        example_passengers.to_dict("records"),
        example_predictions,
        example_probabilities,
    ):
        status = "survived" if prediction == 1 else "did not survive"
        print(f"{passenger} -> {status} ({probability:.2f} survival probability)")


if __name__ == "__main__":
    main()






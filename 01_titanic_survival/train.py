from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# project_folder/data/titanic_tiny.csv, create path to the dataset
# know the address of the file 

DATA_PATH = Path(__file__).parent / "data" / "titanic_tiny.csv"


def main():
# load the data in to a panda object data structure
    data = pd.read_csv(DATA_PATH)   

# features model use for prediction, each effects the probabliity of surviing
    feature_columns = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
# output column
    target_column = "survived"

# select only these columns 
    X = data[feature_columns]
    y = data[target_column]

    numeric_features = ["pclass", "age", "sibsp", "parch", "fare"] 
    categorical_features = ["sex", "embarked"] 

#preprocessing: data can be messy, preprocessing is used to tun messy ddata like text into number and normalise big numbers

    preprocessor = ColumnTransformer(
        transformers=[
            ("numbers", StandardScaler(), numeric_features),
            ("categories", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

#pipeline automatically moves preprocesser to classifier, like a convery belt

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

#split data for training and test 
# random just shuffling cards (data samples)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

#fitting the model in a regression (weights)
    model.fit(X_train, y_train)

#predict the survival rate of test passengers
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("Titanic survival model")
    print(f"Rows: {len(data)}")
    print(f"Accuracy: {accuracy:.2f}")
    print()
    print(classification_report(y_test, predictions, zero_division=0))

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


# 1. Find/load the dataset
#         ↓
# 2. Choose inputs (X) and outputs (y)
#         ↓
# 3. Prepare/preprocess the data
#         ↓
# 4. Build the machine learning pipeline
#         ↓
# 5. Split data into training/testing sets
#         ↓
# 6. Train the model
#         ↓
# 7. Make predictions
#         ↓
# 8. Evaluate accuracy
#         ↓
# 9. Predict new passengers
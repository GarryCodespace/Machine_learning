from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


DATA_PATH = Path(__file__).parent / "data" / "messages_tiny.csv"


def main():
    data = pd.read_csv(DATA_PATH)

    X = data["message"]
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=1,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            ("vectorizer", CountVectorizer()),
            ("classifier", MultinomialNB()),
        ]
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("Spam detector")
    print(f"Rows: {len(data)}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")
    print()
    print("Confusion matrix")
    print(confusion_matrix(y_test, predictions, labels=["ham", "spam"]))
    print()
    print(classification_report(y_test, predictions, zero_division=0))

    example_messages = [
        "free prize click now",
        "can you send me the homework notes",
        "urgent reward available today",
    ]

    example_predictions = model.predict(example_messages)

    print("Example predictions")
    for message, prediction in zip(example_messages, example_predictions):
        print(f"{message!r} -> {prediction}")


if __name__ == "__main__":
    main()

# Spam Detector

Goal: predict whether a short message is spam or not spam.

This is a beginner text classification project. It turns words into counts, then trains a classifier.

## Run

```bash
python 03_spam_detector/train.py
```

## Things To Interrogate

- How can text become numbers?
- What is a bag-of-words model?
- What does `CountVectorizer` do?
- What is Naive Bayes?
- Why might precision and recall matter more than accuracy?

## Modifications

- Add 10 new messages to the CSV.
- Change `CountVectorizer` to include bigrams.
- Try `LogisticRegression`.
- Print the most common words in spam messages.
- Test your own fake spam and real messages.

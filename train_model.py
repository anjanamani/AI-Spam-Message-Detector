from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset
data = pd.read_csv(
    "dataset/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)

# 2. Convert labels into numbers
data["label"] = data["label"].map({
    "ham": 0,
    "spam": 1
})

# 3. Separate messages and labels
X = data["message"]
y = data["label"]

# 4. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. Define three algorithms
algorithms = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(kernel="linear", probability=True)
}

best_model = None
best_score = -1
best_name = ""

# 6. Train and evaluate each algorithm
for name, classifier in algorithms.items():

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
        )),
        ("classifier", classifier)
    ])

    # Train
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(
        y_test,
        predictions,
        target_names=["Normal", "Spam"],
        output_dict=True
    )

    spam_precision = report["Spam"]["precision"]
    spam_recall = report["Spam"]["recall"]
    spam_f1 = report["Spam"]["f1-score"]

    print("\n------------------------------")
    print("Algorithm:", name)
    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("Spam Precision:", round(spam_precision, 3))
    print("Spam Recall:", round(spam_recall, 3))
    print("Spam F1-score:", round(spam_f1, 3))

    # Select the best model using spam F1-score
    if spam_f1 > best_score:
        best_score = spam_f1
        best_model = model
        best_name = name

# 7. Save the best model
joblib.dump(best_model, "spam_model.pkl")

print("\n==============================")
print("BEST MODEL:", best_name)
print("BEST SPAM F1-SCORE:", round(best_score, 3))
print("Best model saved successfully!")
# 8. Generate confusion matrix
best_predictions = best_model.predict(X_test)

cm = confusion_matrix(y_test, best_predictions)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Spam"]
)

display.plot(cmap="Blues")

plt.title("Spam Message Detection - Confusion Matrix")
plt.show()
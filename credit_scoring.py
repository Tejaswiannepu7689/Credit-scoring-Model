import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score

# Load Dataset
data = pd.read_csv("credit_data.csv")

# Features and Target
X = data[['income', 'debt', 'payment_history']]
y = data['credit_score']

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# Models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

# Store Results
results = {
    "Model": [],
    "Accuracy": [],
    "Precision": [],
    "Recall": [],
    "F1 Score": []
}

print("\n===== MODEL RESULTS =====\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results["Model"].append(name)
    results["Accuracy"].append(round(accuracy * 100, 2))
    results["Precision"].append(round(precision * 100, 2))
    results["Recall"].append(round(recall * 100, 2))
    results["F1 Score"].append(round(f1 * 100, 2))

    print(f"{name}")
    print("Accuracy :", round(accuracy * 100, 2), "%")
    print("Precision:", round(precision * 100, 2), "%")
    print("Recall   :", round(recall * 100, 2), "%")
    print("F1 Score :", round(f1 * 100, 2), "%")
    print("-" * 40)

# Create DataFrame
results_df = pd.DataFrame(results)

print("\nPerformance Comparison Table\n")
print(results_df)

# Plot Graph
results_df.set_index("Model").plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Credit Scoring Model Comparison")
plt.xlabel("Models")
plt.ylabel("Score (%)")
plt.ylim(0, 100)
plt.grid(axis='y')
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()
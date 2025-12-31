import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 1. DATA LOADING & PREPROCESSING 
def load_data(path):
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split('\t', 1)
            if len(parts) == 2:
                labels.append(1 if parts[0] == "spam" else 0)
                texts.append(parts[1])
    return texts, np.array(labels)

print("🚀 Loading data for Phase 2...")
# Adjust path if your data is in a different folder
texts, labels = load_data("../Data/spams.txt") 
vectorizer = CountVectorizer(max_features=1000)
X = vectorizer.fit_transform(texts).toarray()
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# 2. NAIVE BAYES FROM SCRATCH 
class MyMultinomialNB:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        self._priors = np.zeros(len(self.classes))
        self._likelihoods = np.zeros((len(self.classes), n_features))

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            # Calculating Priors 
            self._priors[idx] = X_c.shape[0] / n_samples
            # Bayesian Theorem with Laplace Smoothing (+1) 
            self._likelihoods[idx, :] = (X_c.sum(axis=0) + 1) / (X_c.sum() + n_features)

    def predict(self, X):
        return [self._predict_one(x) for x in X]

    def _predict_one(self, x):
        posteriors = []
        for idx, c in enumerate(self.classes):
            # Using Log to prevent underflow in Bayesian calculation 
            prior = np.log(self._priors[idx])
            likelihood = np.sum(np.log(self._likelihoods[idx, :]) * x)
            posteriors.append(prior + likelihood)
        return self.classes[np.argmax(posteriors)]

# 3. TRAINING & EVALUATION
print("📊 Training Naive Bayes (Scratch)...")
nb_scratch = MyMultinomialNB()
nb_scratch.fit(X_train, y_train)
y_pred_nb = nb_scratch.predict(X_test)

# 4. PERFORMANCE METRICS (Requirement: Marks 10) [cite: 48, 49, 50, 51]
metrics = {
    "Accuracy": accuracy_score(y_test, y_pred_nb),
    "Precision": precision_score(y_test, y_pred_nb),
    "Recall": recall_score(y_test, y_pred_nb),
    "F1-Score": f1_score(y_test, y_pred_nb)
}

print("\n--- PHASE 2 PERFORMANCE (NAIVE BAYES SCRATCH) ---")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

# 5. COMPARISON TABLE (Requirement: Deliverable) 
comparison_table = pd.DataFrame({
    "Model": ["Decision Tree (Scratch)", "Naive Bayes (Scratch)"],
    "Accuracy": [0.9426, metrics["Accuracy"]],
    "Precision": [0.8540, metrics["Precision"]],
    "Recall": [0.7267, metrics["Recall"]],
    "F1-Score": [0.7852, metrics["F1-Score"]]
})

print("\n--- COMPARISON WITH DECISION TREE ---")
print(comparison_table.to_string(index=False))

# 6. CONFUSION MATRIX (Requirement: Deliverable) [cite: 53]
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred_nb), annot=True, fmt='d', cmap='Purples',
            xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title("Confusion Matrix: Naive Bayes Scratch")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig("phase2_nb_cm.png")
print("\n✅ SUCCESS: 'phase2_nb_cm.png' saved.")
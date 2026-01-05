import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_curve, auc

# 1. LOAD AND PREPROCESS (Using TF-IDF for better SVM performance)
def load_data(path):
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split('\t', 1)
            if len(parts) == 2:
                labels.append(1 if parts[0] == "spam" else 0)
                texts.append(parts[1])
    return texts, np.array(labels)

print("🚀 Starting Phase 3: Scikit-Learn Comparison...")
texts, labels = load_data("../Data/spams.txt")
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(texts).toarray()
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# 2. NAIVE BAYES (SCIKIT-LEARN)
start_nb = time.time()
nb_sklearn = MultinomialNB()
nb_sklearn.fit(X_train, y_train)
nb_time = time.time() - start_nb
y_score_nb = nb_sklearn.predict_proba(X_test)[:, 1]

# 3. SVM WITH HYPERPARAMETER TUNING & SCALING
# Feature Scaling is required for SVM [Handbook Phase 3]
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

start_svm = time.time()
# Hyperparameter tuning using GridSearchCV [Handbook Phase 3]
param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
svm_grid = GridSearchCV(SVC(probability=True), param_grid, cv=3)
svm_grid.fit(X_train_scaled, y_train)
svm_time = time.time() - start_svm
y_score_svm = svm_grid.predict_proba(X_test_scaled)[:, 1]

# 4. RESULTS TABLE
results = pd.DataFrame({
    "Model": ["Naïve Bayes", "SVM (Tuned)"],
    "Accuracy": [accuracy_score(y_test, nb_sklearn.predict(X_test)), 
                 accuracy_score(y_test, svm_grid.predict(X_test_scaled))],
    "Training Time (s)": [nb_time, svm_time]
})
print("\n--- PHASE 3: ACCURACY & TIME COMPARISON ---")
print(results.to_string(index=False))

# 5. ROC CURVE PLOTS
fpr_nb, tpr_nb, _ = roc_curve(y_test, y_score_nb)
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_score_svm)

plt.figure(figsize=(8, 6))
plt.plot(fpr_nb, tpr_nb, label=f'Naïve Bayes (AUC = {auc(fpr_nb, tpr_nb):.2f})')
plt.plot(fpr_svm, tpr_svm, label=f'SVM (AUC = {auc(fpr_svm, tpr_svm):.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison: NB vs SVM')
plt.legend()
plt.savefig('phase3_roc_curve.png')
print("\n✅ ROC Curve saved as 'phase3_roc_curve.png'")

# 6. SAVE SVM MODEL
import joblib
import os

model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Phase7/models'))
os.makedirs(model_dir, exist_ok=True)

print(f"\n💾 Saving SVM Model to {model_dir}...")
# Refit best estimator on full data or just save the grid search best estimator
best_svm = svm_grid.best_estimator_
joblib.dump(best_svm, os.path.join(model_dir, 'svm_model.pkl'))
joblib.dump(scaler, os.path.join(model_dir, 'phase3_scaler.pkl'))
joblib.dump(vectorizer, os.path.join(model_dir, 'phase3_vectorizer.pkl'))

print("✅ SVM Model saved.")
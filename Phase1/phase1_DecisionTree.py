import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Imports for Comparison
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import lightgbm as lgb

# 1. LOAD DATA
def load_data(path):
    texts, labels = [], []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split('\t', 1)
                if len(parts) == 2:
                    labels.append(1 if parts[0] == "spam" else 0)
                    texts.append(parts[1])
        return texts, np.array(labels)
    except FileNotFoundError:
        print(f"Error: File '../Data/spams.txt' not found!")
        return [], []

print("🚀 Starting Phase 1 Final Execution...")
texts, labels = load_data("../Data/spams.txt")
vectorizer = CountVectorizer(max_features=500)
X = vectorizer.fit_transform(texts).toarray()
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# 2. FROM SCRATCH LOGIC 
def entropy(y):
    if len(y) == 0: return 0
    counts = np.bincount(y); ps = counts / len(y)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])

def info_gain(X_col, y):
    parent_ent = entropy(y)
    vals, counts = np.unique(X_col, return_counts=True)
    weighted_ent = np.sum([(counts[i]/len(y)) * entropy(y[X_col == vals[i]]) for i in range(len(vals))])
    return parent_ent - weighted_ent

class MyDecisionTree:
    def __init__(self, depth=3): self.depth = depth
    def fit(self, X, y, d=0):
        if len(set(y)) == 1 or d == self.depth: return Counter(y).most_common(1)[0][0]
        gains = [info_gain(X[:, i], y) for i in range(X.shape[1])]
        best_feat = np.argmax(gains)
        tree = {best_feat: {}}
        for v in np.unique(X[:, best_feat]):
            tree[best_feat][v] = self.fit(X[X[:, best_feat] == v], y[X[:, best_feat] == v], d + 1)
        return tree
    def predict(self, X, tree):
        def pred_one(x, t):
            if not isinstance(t, dict): return t
            feat = list(t.keys())[0]
            return pred_one(x, t[feat].get(x[feat], 0))
        return np.array([pred_one(x, tree) for x in X])

# 3. TRAINING
print("📊 Training all models including Advanced Ensembles...")
scratch_model = MyDecisionTree(depth=3)
my_tree_logic = scratch_model.fit(X_train, y_train)
y_p_scratch = scratch_model.predict(X_test, my_tree_logic)

# Comparison models
models = {
    "My Scratch Tree": y_p_scratch,
    "DecisionTree (SK)": DecisionTreeClassifier(max_depth=3).fit(X_train, y_train).predict(X_test),
    "Random Forest": RandomForestClassifier().fit(X_train, y_train).predict(X_test),
    "Extra Trees": ExtraTreesClassifier().fit(X_train, y_train).predict(X_test),
    "Gradient Boosting": GradientBoostingClassifier().fit(X_train, y_train).predict(X_test),
    "XGBoost": XGBClassifier().fit(X_train, y_train).predict(X_test),
    "LightGBM": lgb.LGBMClassifier(verbose=-1).fit(X_train, y_train).predict(X_test)
}

# 4. RESULTS TABLE
results = []
for name, y_pred in models.items():
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred)
    })

print("\n--- PHASE 1 PERFORMANCE TABLE ---")
print(pd.DataFrame(results).to_string(index=False))

# 5. VISUALIZATIONS (REQUIRED FOR HANDBOOK)
print("\n🖼️ Generating Visualizations...")

# Confusion Matrix for Scratch Model
cm = confusion_matrix(y_test, y_p_scratch)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title('Confusion Matrix: My Scratch Tree')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png')


# Decision Tree Visualization
plt.figure(figsize=(20,10))
sk_tree = DecisionTreeClassifier(max_depth=3).fit(X_train, y_train)
plot_tree(sk_tree, filled=True, feature_names=vectorizer.get_feature_names_out(), class_names=['Ham','Spam'], fontsize=10)
plt.title("Decision Tree Visualization (Logic Flow)")
plt.savefig('tree_viz.png')


print("✅ SUCCESS!")
print("1. Table printed above.")
print("2. 'confusion_matrix.png' saved.")
print("3. 'tree_viz.png' saved.")
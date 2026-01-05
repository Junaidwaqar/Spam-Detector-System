import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# ------------------ Load Dataset ------------------
def load_data(path):
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split('\t', 1)
            if len(parts) == 2:
                labels.append(1 if parts[0] == "spam" else 0)
                texts.append(parts[1])
    return texts, np.array(labels)

texts, labels = load_data("../Data/spams.txt")

vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(texts).toarray()

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test  = torch.tensor(X_test,  dtype=torch.float32)
y_train = torch.tensor(y_train.reshape(-1,1), dtype=torch.float32)
y_test  = torch.tensor(y_test.reshape(-1,1),  dtype=torch.float32)

# ------------------ Single Layer Network ------------------
class SingleLayerNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1000, 1)   # SINGLE neuron

    def forward(self, x):
        return torch.sigmoid(self.fc(x))

model = SingleLayerNN()

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# ------------------ Training ------------------
losses = []

print("Training Phase-4 Neural Network...\n")

for epoch in range(20):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    print(f"Epoch {epoch+1}/20  Loss: {loss.item():.4f}")

# ------------------ Test Accuracy ------------------
with torch.no_grad():
    predictions = model(X_test)
    predicted = (predictions >= 0.5).float()
    accuracy = (predicted == y_test).sum() / y_test.size(0)

print("\nFinal Test Accuracy:", round(accuracy.item()*100,2), "%")

# ------------------ Convergence Plot ------------------
plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Phase-4 Convergence Plot")
plt.savefig("phase4_convergence.png")
plt.show()

print("\nphase4_convergence.png saved successfully.")

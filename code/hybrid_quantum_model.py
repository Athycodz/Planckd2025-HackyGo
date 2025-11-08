# hybrid_quantum_model.py
import pennylane as qml
from pennylane import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from torchvision import datasets, transforms

# -------------------------
# 1) Load MNIST subset
# -------------------------
transform = transforms.Compose([transforms.ToTensor()])
train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# keep a small subset for speed (you can increase later)
train_items = [(x, y) for (x, y) in zip(train.data, train.targets) if y in [0, 1]][:400]
test_items  = [(x, y) for (x, y) in zip(test.data, test.targets) if y in [0, 1]][:100]

X_train_raw = np.array([x.numpy().flatten().astype(float) for (x, _) in train_items])
Y_train     = np.array([int(y.numpy()) for (_, y) in train_items])
X_test_raw  = np.array([x.numpy().flatten().astype(float) for (x, _) in test_items])
Y_test      = np.array([int(y.numpy()) for (_, y) in test_items])

# -------------------------
# 2) Feature reduction: PCA -> scale to [0,1]
# -------------------------
n_qubits = 6               # increase qubits so circuit has richer input
pca = PCA(n_components=n_qubits)
X_train_pca = pca.fit_transform(X_train_raw)
X_test_pca  = pca.transform(X_test_raw)

scaler = MinMaxScaler(feature_range=(0, 1))
X_train = scaler.fit_transform(X_train_pca)
X_test  = scaler.transform(X_test_pca)

# Convert to pennylane numpy arrays (float64)
X_train = np.array(X_train, requires_grad=False)
X_test  = np.array(X_test, requires_grad=False)
Y_train = np.array(Y_train, requires_grad=False)
Y_test  = np.array(Y_test, requires_grad=False)

# map labels {0,1} to {-1, +1} for regression on PauliZ expectation
Y_train_reg = np.where(Y_train == 1, 1.0, -1.0)
Y_test_reg  = np.where(Y_test  == 1, 1.0, -1.0)

# -------------------------
# 3) Quantum device & circuit
# -------------------------
dev = qml.device("default.qubit", wires=n_qubits, shots=None)

n_layers = 2
# use AngleEmbedding + StronglyEntanglingLayers for expressivity
@qml.qnode(dev)
def quantum_circuit(inputs, weights):
    # encode features as Y-rotations (AngleEmbedding)
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
    # variational entangling layers
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return qml.expval(qml.PauliZ(0))

# weights shape: (n_layers, n_qubits, 3) required by StronglyEntanglingLayers
weight_shape = (n_layers, n_qubits, 3)
# small random init in [-0.1,0.1]
np.random.seed(42)
weights = np.random.normal(scale=0.1, size=weight_shape)
weights = np.array(weights, requires_grad=True)

# -------------------------
# 4) Training settings
# -------------------------
batch_size = 16
opt = qml.AdamOptimizer(stepsize=0.1)
epochs = 25

def batch_iterator(X, Y, batch_size):
    n = len(X)
    idx = np.arange(n)
    np.random.shuffle(idx)
    for i in range(0, n, batch_size):
        batch_idx = idx[i : i + batch_size]
        yield X[batch_idx], Y[batch_idx]

# cost = mean squared error between expectation and target (-1/+1)
def batch_cost(w, Xb, Yb):
    preds = [quantum_circuit(x, w) for x in Xb]
    preds = np.array(preds)
    return np.mean((preds - Yb) ** 2)

# -------------------------
# 5) Training loop
# -------------------------
for epoch in range(epochs):
    epoch_loss = 0.0
    num_batches = 0
    for Xb, Yb in batch_iterator(X_train, Y_train_reg, batch_size):
        # optimizer step on batch
        weights = opt.step(lambda w: batch_cost(w, Xb, Yb), weights)
        batch_loss = batch_cost(weights, Xb, Yb)
        epoch_loss += batch_loss
        num_batches += 1
    epoch_loss /= num_batches
    if (epoch + 1) % 1 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Avg batch loss: {epoch_loss:.4f}")

# -------------------------
# 6) Evaluation
# -------------------------
preds = []
for x in X_test:
    val = quantum_circuit(x, weights)
    preds.append(1 if val > 0 else 0)
acc = accuracy_score(Y_test, preds)
print("Final Quantum Hybrid Accuracy:", acc)

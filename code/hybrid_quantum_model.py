import pennylane as qml
from pennylane import numpy as np
from sklearn.metrics import accuracy_score
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# --- Load a small MNIST subset ---
transform = transforms.Compose([transforms.ToTensor()])
train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# Use only digits 0 and 1 for binary classification
train_data = [(x, y) for (x, y) in zip(train.data, train.targets) if y in [0, 1]]
test_data  = [(x, y) for (x, y) in zip(test.data, test.targets) if y in [0, 1]]

# Convert tensors to numpy and normalize
X_train = np.array([x.numpy().flatten()/255 for (x, _) in train_data[:200]])
Y_train = np.array([y.numpy() for (_, y) in train_data[:200]])
X_test  = np.array([x.numpy().flatten()/255 for (x, _) in test_data[:50]])
Y_test  = np.array([y.numpy() for (_, y) in test_data[:50]])

# --- Quantum layer setup ---
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def quantum_circuit(inputs, weights):
    # Encode first 4 features into rotations
    for i in range(n_qubits):
        qml.RY(np.pi * inputs[i], wires=i)
    # Simple entangling layer
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i+1])
    # Trainable rotations
    for i in range(n_qubits):
        qml.RY(weights[i], wires=i)
    return qml.expval(qml.PauliZ(0))

# Initialize weights
weights = np.random.randn(n_qubits, requires_grad=True)

# --- Hybrid training loop ---
opt = qml.GradientDescentOptimizer(stepsize=0.4)
epochs = 10

for epoch in range(epochs):
    total_loss = 0
    for x, y in zip(X_train, Y_train):
        y_target = 1 if y == 1 else -1
        pred = quantum_circuit(x[:n_qubits], weights)
        loss = (pred - y_target) ** 2
        weights = opt.step(lambda w: (quantum_circuit(x[:n_qubits], w) - y_target)**2, weights)
        total_loss += loss
    print(f"Epoch {epoch+1}, Loss {total_loss/len(X_train):.4f}")

# --- Evaluation ---
preds = [1 if quantum_circuit(x[:n_qubits], weights) > 0 else 0 for x in X_test]
acc = accuracy_score(Y_test, preds)
print("Quantum Hybrid Accuracy:", acc)

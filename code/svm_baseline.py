from sklearn import svm, metrics
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Load MNIST
transform = transforms.Compose([transforms.ToTensor()])
train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# Convert to DataLoader
train_loader = DataLoader(train, batch_size=2000, shuffle=True)
test_loader = DataLoader(test, batch_size=2000, shuffle=False)

# Take first batch only (SVM is slow on full dataset)
x_train, y_train = next(iter(train_loader))
x_test, y_test = next(iter(test_loader))

# Flatten images: [N, 1, 28, 28] → [N, 784]
x_train = x_train.view(x_train.size(0), -1)
x_test = x_test.view(x_test.size(0), -1)

# Train SVM
clf = svm.SVC(kernel='linear')
clf.fit(x_train, y_train)

# Predict
y_pred = clf.predict(x_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

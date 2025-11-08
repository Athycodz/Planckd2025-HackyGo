import torchvision
from torchvision import transforms

mnist = torchvision.datasets.MNIST(root='./data', download=True, train=True, transform=transforms.ToTensor())
print("Samples:", len(mnist))
img, label = mnist[0]
print("Image tensor shape:", img.shape, "Label:", label)

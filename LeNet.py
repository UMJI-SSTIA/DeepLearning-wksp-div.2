# Reference: JI DeepLearning Tutorial, Dive into deep learning
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

trainset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
testset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

class MNIST_LeNEt(nn.Module):
    def __init__(self):
        super(MNIST_LeNEt, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6,kernel_size=5,padding=2)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5) 
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)  
        self.fc2 = nn.Linear(in_features=120, out_features=84)    
        self.fc3 = nn.Linear(in_features=84,out_features=10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = x.view(-1, 16 * 5 * 5)  # Flatten the tensor
        # x = F.sigmoid(self.fc1(x))
        x = F.relu(self.fc1(x))
        # x = F.sigmoid(self.fc2(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  # visualize the predict result
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
        # tensor
            ax.imshow(img.numpy())
        else:
        # PIL figures
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    plt.show()
    return axes

def get_mnist_labels(labels): 
    figure_labels = ['0','1', '2', '3', '4', '5','6', '7', '8', '9']
    return [figure_labels[int(i)] for i in labels]

model = MNIST_LeNEt()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training Loop
for epoch in range(5):
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Print loss
        if i % 500 == 0:
            print(
                f"Epoch [{epoch + 1}/5], Step [{i + 1}/{len(trainloader)}], Loss: {loss.item()}"
            )

correct = 0
total = 0
num_image_shown = 10
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
print(f"Accuracy: {100 * correct / total}%")

show_images(images[0:num_image_shown].reshape((num_image_shown,28,28)),
                    1,num_image_shown,titles=get_mnist_labels(predicted))
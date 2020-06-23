# based of https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
from model import CustomModel
import torch.optim as optim

n_epochs = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # if GPU available, use it

image_res = (240, 320)

data_transforms = transforms.Compose([
    transforms.Resize(image_res),
    transforms.ToTensor()])  # transforms to resize and convert the image to a tensor object

data_dir = 'images'  # folders with all the images separated by class in subfolders
dataset = datasets.ImageFolder(data_dir, data_transforms)
img_loader = DataLoader(dataset, batch_size=4, shuffle=True)  # separates the dataset in batches for training
# the batch size may be limited by memory. Reduce it if that is the case

n_classes = 2  # classify 2 different images
net = CustomModel(image_res, n_classes).to(device)  # custom model initialization

optimizer = optim.Adam(net.parameters(), lr=0.001)  # optimizer (basically a gradient descent)

loss = torch.nn.CrossEntropyLoss()  # loss function for classification

for epoch in range(n_epochs):

    epoch_loss = 0.0
    print(epoch)
    for i, data in enumerate(img_loader, 0):  # iterator to get the batches for training

        inputs, labels = data

        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)

        optimizer.zero_grad()  # reset gradients for a new calculation

        loss_size = loss(outputs, labels)

        loss_size.backward()  # back-propagation of the loss

        epoch_loss += loss_size.data

        optimizer.step()  # optimizer step based on the back-propagation results

net.eval()  # freezes the layers from learning to evaluate the network
correct = 0
total = 0

for i, data in enumerate(img_loader, 0):
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = net(inputs)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)  # get the total number of images
    correct += (predicted == labels).sum().item()  # get if the prediction is correct

print("Accuracy: {:2f}".format(100 * correct / total))

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metric

import os

'''
This code is adapted from two sources:
(i) The official PyTorch MNIST example (https://github.com/pytorch/examples/blob/master/mnist/main.py)
(ii) Starter code from Yisong Yue's CS 155 Course (http://www.yisongyue.com/courses/cs155/2020_winter/)
'''



class fcNet(nn.Module):
    '''
    Design your model with fully connected layers (convolutional layers are not
    allowed here). Initial model is designed to have a poor performance. These
    are the sample units you can try:
        Linear, Dropout, activation layers (ReLU, softmax)
    '''
    def __init__(self):
        # Define the units that you will use in your model
        # Note that this has nothing to do with the order in which operations
        # are applied - that is defined in the forward function below.
        super(fcNet, self).__init__()
        self.fc1 = nn.Linear(in_features=784, out_features=20)
        self.fc2 = nn.Linear(20, 10)
        self.dropout1 = nn.Dropout(p=0.5)

    def forward(self, x):
        # Define the sequence of operations your model will apply to an input x
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = F.relu(x)

        output = F.log_softmax(x, dim=1)
        return output


class ConvNet(nn.Module):
    '''
    Design your model with convolutional layers.
    '''
    # def __init__(self):
    #     super(ConvNet, self).__init__()
    #     self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=1)
    #     self.conv2 = nn.Conv2d(8, 8, 3, 1)
    #     self.dropout1 = nn.Dropout2d(0.5)
    #     self.dropout2 = nn.Dropout2d(0.5)
    #     self.fc1 = nn.Linear(200, 64)
    #     self.fc2 = nn.Linear(64, 10)

    # def forward(self, x):
    #     x = self.conv1(x)
    #     x = F.relu(x)
    #     x = F.max_pool2d(x, 2)
    #     x = self.dropout1(x)

    #     x = self.conv2(x)
    #     x = F.relu(x)
    #     x = F.max_pool2d(x, 2)
    #     x = self.dropout2(x)

    #     x = torch.flatten(x, 1)
    #     x = self.fc1(x)
    #     x = F.relu(x)
    #     x = self.fc2(x)

    #     output = F.log_softmax(x, dim=1)
    #     return output
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(5,5), stride=1, padding=3, padding_mode='circular')
        self.conv2 = nn.Conv2d(8, 12, 3, stride=1, padding=2, padding_mode='circular')
        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(432, 64)
        self.fc2 = nn.Linear(64, 10)
        #self.fc3 = nn.Linear(64, 10)
        self.bnorm1 = nn.BatchNorm2d(8)
        self.bnorm2 = nn.BatchNorm2d(12)

    def forward(self, x): # 28x28x1
        x = self.conv1(x) # 28x28x8
        x = self.bnorm1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2) #14x14x8
        #x = self.dropout1(x)

        x = self.conv2(x) #13x13x12
        x = self.bnorm2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2) #6x6x12
        #x = self.dropout2(x)

        x = torch.flatten(x, 1) #432
        x = self.fc1(x) #200
        x = F.relu(x)
        x = self.fc2(x) #64

        output = F.log_softmax(x, dim=1)
        return output


class Net(nn.Module):
    '''
    Build the best MNIST classifier.
    '''
    def __init__(self):
        super(Net, self).__init__()

    def forward(self, x):
        return x


def train(args, model, device, train_loader, optimizer, epoch):
    '''
    This is your training function. When you call this function, the model is
    trained for 1 epoch.
    '''
    model.train()   # Set the model to training mode
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()               # Clear the gradient
        output = model(data)                # Make predictions
        loss = F.nll_loss(output, target)   # Compute loss
        loss.backward()                     # Gradient computation
        optimizer.step()                    # Perform a single optimization step
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.sampler),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()    # Set the model to inference mode
    test_loss = 0
    correct = 0
    test_num = 0
    n_show = 0;
    show_confusion = False;
    all_preds = []
    all_targets = []
    with torch.no_grad():   # For the inference step, gradient is not computed
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            test_num += len(data)
            # Visualize
            rng = np.arange(len(data))
            correct_preds = pred.eq(target.view_as(pred)).numpy().reshape(len(data))
            correct_inds = rng[correct_preds == False]
            
            all_targets.extend(target.numpy().tolist())
            all_preds.extend(pred.numpy().reshape(len(pred)).tolist())
            
            for ind in correct_inds:
                if (n_show > 0):
                    plt.imshow(data[ind].numpy().reshape((28,28)), cmap='gray')
                    plt.show() 
                    print(pred[ind])
                    print(target[ind])
                    n_show = n_show-1
    if show_confusion:
            labels = ['0','1','2','3','4','5','6','7','8','9']
            cm = metric.confusion_matrix(all_targets, all_preds)/1100
            fig = plt.figure()
            ax = fig.add_subplot(111)
            cax = ax.matshow(cm)
            plt.title('Confusion matrix of the classifier')
            fig.colorbar(cax)
            ax.set_xticklabels([''] + labels)
            ax.set_yticklabels([''] + labels)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.show()
    test_loss /= test_num

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, test_num,
        100. * correct / test_num))


def main():
    # Training settings
    # Use the command line to modify the default settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--step', type=int, default=1, metavar='N',
                        help='number of epochs between learning rate reductions (default: 1)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='evaluate your model on the official test set')
    parser.add_argument('--load-model', type=str,
                        help='model file path')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}



        
    # Evaluate on the official test set
    if args.evaluate:
        assert os.path.exists(args.load_model)

        # Set the test model
        model = ConvNet().to(device)
        model.load_state_dict(torch.load(args.load_model))
        
        show_conv = False
        if show_conv:
            # Visualize conv filter
            kernels = model.conv1.weight.detach()
            for idx in range(9):
                plt.imshow(kernels[idx].squeeze(), cmap='gray')
                plt.show()

        test_dataset = datasets.MNIST('../data', train=False,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)

        test(model, device, test_loader)

        return

    # Pytorch has default MNIST dataloader which loads data at each iteration
    train_dataset = datasets.MNIST('../data', train=True, download=True,
                transform=transforms.Compose([       # Data preprocessing
                    #transforms.RandomResizedCrop(28, scale=(0.5, 1.0)),
                    # transforms.RandomPerspective(),
                    transforms.ToTensor(),           # Add data augmentation here
                    transforms.Normalize((0.1307,), (0.3081,))
                ]))

    # You can assign indices for training/validation or use a random subset for
    # training by using SubsetRandomSampler. Right now the train and validation
    # sets are built from the same indices - this is bad! Change it so that
    # the training and validation sets are disjoint and have the correct relative sizes.
    np.random.seed(2021)
    ndata = len(train_dataset)
    valid_ratio = 0.15
    subset_ratio = 1 # Discards a ratio of the full training data for testing purposes. this ratio is the number to keep
    subset_indices_valid = [];
    subset_indices_train = [];
    # Splits the dataset per class
    for n in range(0,10):
        # extract the indices of the class
        class_ind_bool = train_dataset.targets==n
        class_ind_bool = class_ind_bool.numpy()
        rng = np.arange(ndata)
        class_idx = rng[class_ind_bool]
        # Orders the indices of the class randomly
        np.random.shuffle(class_idx)
        nkeep = int(subset_ratio * len(class_idx))
        # Draws the first valid_ratio of indices as the validation set
        nvalid = int(valid_ratio * nkeep)
        class_valid_idx, class_train_idx = class_idx[:nvalid], class_idx[nvalid:nkeep]
        # appends to overall validation/train set. must convert to list first
        subset_indices_valid.extend(class_valid_idx.tolist())
        subset_indices_train.extend(class_train_idx.tolist())

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=SubsetRandomSampler(subset_indices_train)
    )
    val_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.test_batch_size,
        sampler=SubsetRandomSampler(subset_indices_valid)
    )

    # Load your model [fcNet, ConvNet, Net]
    model = ConvNet().to(device)

    # Try different optimzers here [Adam, SGD, RMSprop]
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    # Set your learning rate scheduler
    scheduler = StepLR(optimizer, step_size=args.step, gamma=args.gamma)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, val_loader)
        scheduler.step()    # learning rate scheduler

    test(model, device, train_loader)
    # You may optionally save your model at each epoch here
    if args.save_model:
        torch.save(model.state_dict(), "mnist_model.pt")


if __name__ == '__main__':
    main()

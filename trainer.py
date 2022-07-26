import torch
import keras

class Trainer():
    """
    Class used to train a FCNN model
    """
    def __init__(self, model, device, loss_function, optimizer):
        """
        :param model: Model to train. Needs to inherit torch.nn.Module
        :param device: torch.device. Use 'cuda' for faster computations if possible
        :param loss_function: Loss function to train with (best is to functions from torch.nn)
        :param optimizer: Optimizer to train with. Use torch.optim for it
        """
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.model = model
        self.device = device
        self.train_losses = []
        self.train_accuracies = []

    def train(self, num_epochs, train_loader):
        """
        Trains the model, with regular backpropagation

        :param num_epochs: Number of epochs to run
        :param train_loader: Dataloader of type torch.utils.data.Dataloader for training.

        :return: None
        """
        for epoch in range(num_epochs):
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}]...')
            n_correct = 0
            total_loss = 0
            for i, (data, labels) in enumerate(train_loader):
                # Forward pass
                labels = labels.to(self.device)
                predictions, loss_val = self.forward(data, labels)

                # Backward pass
                self.backward(loss_val)

                _, idxs = torch.max(predictions, 1)
                prediction_label = torch.tensor(keras.utils.to_categorical(idxs.cpu(), num_classes=len(labels[0]))).to(self.device)
                n_correct += 0.5*(prediction_label == labels).sum().item()
                total_loss += loss_val.item()

            train_acc = n_correct / (len(train_loader) * train_loader.batch_size)
            train_loss = total_loss/len(train_loader)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)

    def forward(self, data, labels):
        """
        Forward pass of the data.
        :param data: Inputs to the model
        :param labels: True labels
        :return: (model predictions, loss on the data)
        """
        #data = data.to(self.device)
        predictions = self.model(data).to(self.device)
        loss_val = self.loss_function(predictions, labels)
        return predictions, loss_val

    def backward(self, loss_value):
        """
        Backwards pass for the backprop
        :param loss_value: value of the loss
        :return: None
        """
        loss_value.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def get_data(self):
        """
        Returns data of training
        :return: train_losses, train_acc
        """
        return self.train_losses, self.train_accuracies
    
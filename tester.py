import torch

class Tester():
    def __init__(self, model, device, loss_function):
        self.model = model
        self.device = device
        self.loss_function = loss_function
        self.test_losses = []
        self.test_accuracies = []

    def test(self, test_loader):
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            for data, labels in test_loader:
                data = data.reshape(-1, self.model.input_size).to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(data)
                loss_val = self.loss_function(outputs, labels)
                _, prediction_label = torch.max(outputs, 1)
                n_samples += labels.shape[0]
                n_correct += (prediction_label == labels).sum().item()

            acc = 100.0 * n_correct / n_samples
            return acc, loss_val.item()
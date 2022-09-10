import numpy as np
import torch
import keras
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score


class Tester():
    def __init__(self, model, device, loss_function):
        self.model = model
        self.device = device
        self.loss_function = loss_function

    def test(self, test_loader):
        """
        Tests the model, given a test DataLoader
        :param test_loader: A DataLoader for test data.
        :return: (acc, loss) - accuracy over the test data, and total loss accumulated.
        """
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            total_loss = 0
            for i, (data, labels) in enumerate(test_loader):
                predictions = self.model(data).to(self.device)
                labels = labels.to(self.device)
                total_loss += self.loss_function(predictions, labels).item()
                _, idxs = torch.max(predictions, 1)
                prediction_label = torch.tensor(keras.utils.np_utils.to_categorical(idxs.cpu(), num_classes=len(labels[0]))).to(self.device)
                n_samples += labels.shape[0]
                n_correct += 0.5*(prediction_label == labels).sum().item()

            acc = n_correct / n_samples
            return total_loss, acc


def test_bce_model(model, device, test_dataloader):
    loss_func = torch.nn.BCELoss()
    tester = Tester(model, device, loss_func)
    return tester.test(test_dataloader)


def test_tabnet_model(clf, settings, test_df):
    dataset_name = settings['dataset_name']
    features = settings['FeatureNames']
    target = settings['Target']
    X_test = test_df[features].values
    y_test = test_df[target].values
    preds = np.vstack([x.cpu().detach().numpy() for x in clf.predict_proba(X_test)])
    test_auc = roc_auc_score(y_score=preds[:,1], y_true=y_test)
    print(f"FINAL TEST roc_auc_score FOR {dataset_name} : {test_auc}")
    preds = np.argmax(preds, 1)
    test_acc = accuracy_score(y_pred=preds, y_true=y_test)
    print(f"FINAL TEST accuracy_score FOR {dataset_name} : {test_acc}")
    test_bal_acc = balanced_accuracy_score(y_pred=preds, y_true=y_test)
    print(f"FINAL TEST balanced_accuracy_score FOR {dataset_name} : {test_bal_acc}")
    return test_acc

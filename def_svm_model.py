import numpy as np
import pandas as pd
import torch
import tqdm
from tqdm import tqdm


class SvmModel:
    def __init__(self, settings, model, svm_disc):
        self.settings = settings
        self.model = model
        self.svm_disc = svm_disc

    def classify(self, sample):
        model_predict = self.model(torch.tensor(sample))
        label = model_predict.max(0, keepdim=True)[1].cpu().numpy()[0]

        is_adv = self.svm_disc.predict(pd.DataFrame(sample).transpose())
        if is_adv:
            label = 1 - label

        return label

    def test(self, test_df):
        n_correct = 0
        n_samples = 0
        for _, row in tqdm(test_df.iterrows(), total=test_df.shape[0], desc="{}".format("SVM Model Test")):
            x = row[self.settings['FeatureNames']]
            target = row[self.settings['Target']]
            prediciton = self.classify(x)
            if prediciton == target:
                n_correct +=1
            n_samples += 1
        acc = n_correct / n_samples
        return  acc


class TabnetSvmModel:
    def __init__(self, settings, model, svm_disc):
        self.settings = settings
        self.model = model
        self.svm_disc = svm_disc

    def classify(self, sample):
        x = np.expand_dims(sample.values, axis=0)
        x = torch.FloatTensor(x).to(self.model.device)
        model_predict = self.model.predict_proba_2(x)[0]
        label = model_predict.max(0, keepdim=True)[1].cpu().numpy()

        is_adv = self.svm_disc.predict(pd.DataFrame(sample).transpose())
        if is_adv:
            label = 1 - label

        return label

    def test(self, test_df):
        n_correct = 0
        n_samples = 0
        for _, row in tqdm(test_df.iterrows(), total=test_df.shape[0], desc="{}".format("SVM Model Test"), disable=False):
            x = row[self.settings['FeatureNames']]
            target = row[self.settings['Target']]
            prediciton = self.classify(x)
            if prediciton == target:
                n_correct +=1
            n_samples += 1
        acc = n_correct / n_samples
        return  acc
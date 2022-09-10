import keras
import pandas as pd
from sklearn import svm
from torch.utils.data import DataLoader

from adverse import gen_adv
from adverse_tabnet import gen_adv as tabnet_gen_adv
import numpy as np

from tabular_dataset import TabularDataset


class SVMDiscriminator:
    def __init__(self, settings, model, adv_method, is_weighted=True, c=1.0, kernel='rbf', degree=3, gamma='scale'):
        self.clf = svm.SVC(C=c, kernel=kernel, degree=degree, gamma=gamma)
        self.settings = settings
        self.model = model
        self.adv_method = adv_method
        self.is_weighted = is_weighted

    def train(self, samples_df):
        orig_examples, df, s_rate, pert_norms, w_pert_norms = gen_adv(self.model, self.settings, self.adv_method, samples_df, n=10)
        assert not df.empty
        orig_examples, df, s_rate, pert_norms_2, w_pert_norms_2 = gen_adv(self.model, self.settings, self.adv_method, df)
        assert not df.empty
        if self.is_weighted:
            norm_samples = np.concatenate((w_pert_norms, w_pert_norms_2))
        else:
            norm_samples = np.concatenate((pert_norms, pert_norms_2))
        labels = np.concatenate((np.zeros(len(pert_norms)), np.ones(len(pert_norms_2))))
        norm_samples = norm_samples.reshape(-1, 1)
        self.clf.fit(norm_samples, labels)

    def predict(self, sample):
        """
        Predict is the sample adversarial or not. 1 - is adv. 0 - not.
        :param sample: A single tabular entry
        :return: Prediction is the sample adversarial or not
        """
        orig_examples, df, s_rate, pert_norms, w_pert_norms = gen_adv(self.model, self.settings, self.adv_method, sample, progress=False)
        if self.is_weighted:
            norm_samples = w_pert_norms
        else:
            norm_samples = pert_norms
        if df.empty:
            return np.array([0])
        return self.clf.predict(np.array(norm_samples).reshape(-1, 1))

    def predict_multilpe(self, samples):
        results = []
        for _, row in samples.iterrows():
            results.append(self.predict(pd.DataFrame(row).transpose()))
        return np.array(results).reshape(-1, 1)

    def test(self, true_data_df, adv_data_df):
        """
        Test the accuracy of SVM (Sort of validation test)
        :param test_loader: pytorch DataLoader for test data. Contains real and adv. examples.
        :return: Success rate of the SVM on real data, and on adv data.
        """

        real_predicts = self.predict_multilpe(true_data_df) # Should be all 0.
        adv_predicts = self.predict_multilpe(adv_data_df) # Should be all 1.

        n_correct_real = np.count_nonzero(real_predicts == 0)
        n_correct_adv = np.count_nonzero(adv_predicts == 1)
        n_real = len(real_predicts)
        n_adv = len(adv_predicts)

        return (n_correct_real/n_real), (n_correct_adv/n_adv)


class TabnetSVMDiscriminator(SVMDiscriminator):
    def __init__(self, settings, model, adv_method, is_weighted=True, c=1.0, kernel='rbf', degree=3, gamma='scale'):
        super().__init__(settings, model, adv_method, is_weighted, c, kernel, degree, gamma)

    def train(self, samples_df):
        orig_examples, df, s_rate, pert_norms, w_pert_norms = tabnet_gen_adv(self.model, self.settings,
                                                                             self.adv_method,
                                                                             samples_df, n=10, progress=False)
        assert not df.empty
        orig_examples, df, s_rate, pert_norms_2, w_pert_norms_2 = tabnet_gen_adv(self.model, self.settings,
                                                                                 self.adv_method, df, progress=False)
        assert not df.empty
        if self.is_weighted:
            norm_samples = np.concatenate((w_pert_norms, w_pert_norms_2))
        else:
            norm_samples = np.concatenate((pert_norms, pert_norms_2))
        labels = np.concatenate((np.zeros(len(pert_norms)), np.ones(len(pert_norms_2))))
        norm_samples = norm_samples.reshape(-1, 1)
        self.clf.fit(norm_samples, labels)

    def predict(self, sample):
        """
        Predict is the sample adversarial or not. 1 - is adv. 0 - not.
        :param sample: A single tabular entry
        :return: Prediction is the sample adversarial or not
        """
        orig_examples, df, s_rate, pert_norms, w_pert_norms = tabnet_gen_adv(self.model, self.settings, self.adv_method, sample, progress=False)
        if self.is_weighted:
            norm_samples = w_pert_norms
        else:
            norm_samples = pert_norms
        if df.empty:
            return np.array([0])
        return self.clf.predict(np.array(norm_samples).reshape(-1, 1))


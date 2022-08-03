import numpy as np
import torch.nn
import matplotlib.pyplot as plt

import dataset_factory


from torch.utils.data import DataLoader

from model import Net
from tabular_dataset import TabularDataset
from trainer import Trainer
from tester import Tester
from adverse import gen_adv

from IPython.display import display

SEED = 0


def main():
    # + configurations for adversarial generation
    settings = {'batch_size'   : 100,
                'epochs'       : 100,
                'hidden_dim'   : 100,
                'layers'       : 5,
                'lr'           : 1e-4,
                'MaxIters'     : 20000,
                'Alpha'        : 0.001,
                'Lambda'       : 8.5
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    credit_g = 'credit-g'
    credit_g_train, credit_g_test = dataset_factory.get_train_test_dataset(settings, credit_g, test_size=300)


    train_dataloader = DataLoader(credit_g_train, batch_size=settings['batch_size'], shuffle=True)
    test_dataloader = DataLoader(credit_g_test, batch_size=settings['batch_size'], shuffle=True)
    d_in, d_out = credit_g_train.get_dimensions()

    nn_model = Net(d_in, settings['hidden_dim'], d_out, settings['layers'])

    settings["Model"] = nn_model

    loss_func = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(nn_model.parameters(), lr=settings['lr'])

    trainer = Trainer(nn_model, device, loss_func, optimizer)
    trainer.train(settings['epochs'], train_dataloader)
    train_losses, train_accuracies = trainer.get_data()

    # tester = Tester(nn_model, device, loss_func)
    # test_acc= tester.test(test_dataloader)
    # print(f'Accuracy of the network on test set: {test_acc} %')

    fig, axs = plt.subplots(2, 2)
    x = np.arange(settings['epochs'])
    axs[0, 0].plot(x, train_losses, label="train_losses")
    axs[0, 0].set_title("train_losses")
    axs[0, 1].plot(x, train_accuracies, label="train_accuracies")
    axs[0, 1].set_title("train_accuracies")
    
    # Sub sample
    settings['TestData'] = settings['TestData'].sample(n=2, random_state = SEED)
    #display(settings['TestData'])
    # Generate adversarial examples
    df_adv_lpf = gen_adv(settings, 'LowProFool')
    #display(df_adv_lpf)
    df_adv_df = gen_adv(settings, 'Deepfool')
    settings['AdvData'] = {'LowProFool' : df_adv_lpf, 'Deepfool' : df_adv_df}

    # Fine-tune model using LPF examples
    ft_dataset = TabularDataset(df_adv_lpf, settings['FeatureNames'], settings['Target'], True)
    ft_dataloader = DataLoader(ft_dataset, batch_size=1, shuffle=True)

    ft_loss_func = torch.nn.BCELoss()
    ft_optimizer = torch.optim.Adam(nn_model.parameters(), lr=0.1*settings['lr'])

    ft_trainer = Trainer(nn_model, device, ft_loss_func, ft_optimizer)
    ft_trainer.train(settings['epochs']//10, ft_dataloader)
    ft_train_losses, ft_train_accuracies = ft_trainer.get_data()

    x = np.arange(settings['epochs']//10)
    axs[1, 0].plot(x, ft_train_losses, label="ft_train_losses")
    axs[1, 0].set_title("ft_train_losses")
    axs[1, 1].plot(x, ft_train_accuracies, label="ft_train_accuracies")
    axs[1, 1].set_title("ft_train_accuracies")
    plt.show()

    pass




if __name__ == "__main__":
    main()

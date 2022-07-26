import numpy as np
import torch.nn
import matplotlib.pyplot as plt

import dataset_factory


from torch.utils.data import DataLoader

from model import Net
from trainer import Trainer
from adverse import gen_adv

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
    credit_g_train, credit_g_test= dataset_factory.get_train_test_dataset(settings, credit_g, test_size=300)


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

    fig, axs = plt.subplots()
    x = np.arange(settings['epochs'])
    axs.plot(x, train_losses, label="train_losses")
    axs.plot(x, train_accuracies, label="train_accuracies")
    axs.legend()
    plt.show()

    # Sub sample
    settings['TestData'] = settings['TestData'].sample(n=10, random_state = SEED)

    # Generate adversarial examples
    df_adv_lpf = gen_adv(settings, 'LowProFool')
    df_adv_df = gen_adv(settings, 'Deepfool')
    settings['AdvData'] = {'LowProFool' : df_adv_lpf, 'Deepfool' : df_adv_df}
    
    pass




if __name__ == "__main__":
    main()

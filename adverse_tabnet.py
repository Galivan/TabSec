# Misc
import numpy
import numpy as np
import pandas as pd

# Pytorch
import torch
import torch.nn as nn
from torch.autograd import Variable

import tqdm
from tqdm import tqdm
from tqdm import tqdm_notebook


def gen_adv(model, config, method, df_test, n=-1):
    """
    Generate adversarial examples from given data, using a specific method
    :param model: NN Model we want to fool
    :param config: General configuration
    :param method: Adversarial method name
    :param df_test: Data to make adversarial examples from
    :param n: number of examples to generate
    :return: Dataframe of adversarial examples with original target
             Success rate of the method - adversarial/total
             Norms of perturbations
             Weighted norms of perturbations
    """
    extra_cols = ['orig_pred', 'adv_pred', 'iters']
    feature_names = config['FeatureNames']
    weights = config['Weights']
    bounds = config['Bounds']
    maxiters = config['MaxIters']
    alpha = config['Alpha']
    lambda_ = config['Lambda']
    target = config['Target']
    results = []
    benigns = []
            
    i = -1
    n_samples = 0
    n_success = 0
    pert_norms = []
    total_loop_change = 0
    weighted_pert_norms = []
    for _, row in tqdm(df_test.iterrows(), total=df_test.shape[0], desc="{}".format(method)):
        i += 1
        x = row[feature_names].values
        n_samples += 1
        
        if method == 'LowProFool':
            orig_pred, adv_pred, x_adv, loop_i = lowProFool(x, model, weights, bounds,
                                                            maxiters, alpha, lambda_)
        elif method == 'Deepfool':
            orig_pred, adv_pred, x_adv, loop_i = deepfool(x, model, maxiters, alpha,
                                                          bounds, weights=[])
        else:
            raise Exception("Invalid method", method)

        total_loop_change += loop_i
        pert = x_adv - x

        if orig_pred.item() != adv_pred.item():
            n_success += 1
            pert_norms.append(np.linalg.norm(pert))
            weighted_pert_norms.append(np.linalg.norm(weights * pert))
            results.append(np.array(np.append(x_adv, orig_pred), dtype=np.float32))
            benigns.append(row)
        if n_success == n:
            break

    df = pd.DataFrame(results, columns=feature_names + [target], dtype=np.float32)
    bening_df = pd.DataFrame(benigns, columns=feature_names + [target], dtype=np.float32)
    return bening_df, df, n_success/n_samples, pert_norms, weighted_pert_norms




# Clipping function
def clip(current, low_bound, up_bound):
    """
    Clip the data to be within the natural bounds.
    :param current: Current values of params
    :param low_bound: Lower bound on each param
    :param up_bound: Upper bound on each param
    :return: List of clipped values
    """
    assert(len(current) == len(up_bound) and len(low_bound) == len(up_bound))
    low_bound = torch.FloatTensor(low_bound).to(current.device)
    up_bound = torch.FloatTensor(up_bound).to(current.device)
    clipped = torch.max(torch.min(current, up_bound), low_bound)
    return clipped


def lowProFool(x, model, weights, bounds, maxiters, alpha, lambda_):
    """
    Generates an adversarial examples x' from an original sample x

    :param x: tabular sample
    :param model: neural network
    :param weights: feature importance vector associated with the dataset at hand
    :param bounds: bounds of the datasets with respect to each feature
    :param maxiters: maximum number of iterations ran to generate the adversarial examples
    :param alpha: scaling factor used to control the growth of the perturbation
    :param lambda_: trade off factor between fooling the classifier and generating imperceptible adversarial example
    :return: original label prediction, final label prediction, adversarial examples x', iteration at which the class changed
    """

    r = torch.FloatTensor(1e-4 * np.ones(x.shape))
    r = r.to(model.device)
    r.requires_grad_()
    r.retain_grad()
    v = torch.FloatTensor(np.array(weights)).to(model.device)

    x_into_model = np.expand_dims(x, axis=0)
    x = torch.FloatTensor(x_into_model).to(model.device)
    x = Variable(x, requires_grad=True)
    output = model.predict_proba_2(x)[0]
    orig_pred = output.max(0, keepdim=True)[1].to('cpu').numpy()
    target_pred = np.abs(1 - orig_pred)

    target = np.array([0., 1.], dtype=np.float32) if target_pred == 1 else np.array([1., 0.], dtype=np.float32)
    target = Variable(torch.tensor(target, requires_grad=False)).to(model.device)

    lambda_ = torch.tensor([lambda_]).to(model.device)

    bce = nn.BCELoss()
    l1 = lambda v, r: torch.sum(torch.abs(v * r)) #L1 norm
    l2 = lambda v, r: torch.sqrt(torch.sum(torch.mul(v * r,v * r))) #L2 norm

    best_norm_weighted = np.inf
    best_pert_x = x

    loop_i, loop_change_class = 0, 0
    while loop_i < maxiters:

        if r.grad is not None:
            r.grad.zero_()

        # Computing loss
        loss_1 = bce(output, target)
        loss_2 = l2(v, r)
        loss = (loss_1 + lambda_ * loss_2)

        # Get the gradient
        loss.backward(retain_graph=True)
        grad_r = r.grad.data.cpu().numpy().copy()

        # Guide perturbation to the negative of the gradient
        ri = - grad_r

        # limit huge step
        ri *= alpha

        # Adds new perturbation to total perturbation
        r = r.clone().detach().cpu().numpy() + ri

        # For later computation
        r_norm_weighted = np.sum(np.abs(r * weights))

        # Ready to feed the model
        r = Variable(torch.FloatTensor(r), requires_grad=True)
        r = r.to(model.device)
        r.requires_grad_()
        r.retain_grad()
        # Compute adversarial example
        xprime = x.to(r.device) + r

        # Clip to stay in legitimate bounds
        xprime = clip(xprime[0], bounds[0], bounds[1]).unsqueeze(0)

        # Classify adversarial example
        x = Variable(xprime)
        output = model.predict_proba_2(x)[0]
        output_pred = output.max(0, keepdim=True)[1].to('cpu').numpy()

        # Keep the best adverse at each iterations
        if output_pred != orig_pred and r_norm_weighted < best_norm_weighted:
            best_r = r
            best_norm_weighted = r_norm_weighted
            best_pert_x = xprime

        if output_pred == orig_pred:
            loop_change_class += 1

        loop_i += 1

    # Clip at the end no matter what
    best_pert_x = clip(best_pert_x[0], bounds[0], bounds[1])
    output = model.predict_proba_2(best_pert_x.unsqueeze(0))[0]
    output_pred = output.max(0, keepdim=True)[1].to('cpu').numpy()
    return orig_pred, output_pred, best_pert_x.clone().detach().cpu().numpy(), loop_change_class

# Forked from https://github.com/LTS4/DeepFool
def deepfool(x_old, net, maxiters, alpha, bounds, weights=[], overshoot=0.002):
    """
    :param image: tabular sample
    :param net: network
    :param maxiters: maximum number of iterations ran to generate the adversarial examples
    :param alpha: scaling factor used to control the growth of the perturbation
    :param bounds: bounds of the datasets with respect to each feature
    :param weights: feature importance vector associated with the dataset at hand
    :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
    :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """

    input_shape = x_old.shape
    x = np.expand_dims(x_old, axis=0)
    x_old = torch.tensor(x)
    x = torch.FloatTensor(x).to('cuda')
    x = Variable(x, requires_grad=True)
    output = net.predict_proba_2(x)[0]

    orig_pred = output.max(0, keepdim=True)[1].to('cpu') # get the index of the max log-probability

    origin = Variable(torch.tensor([orig_pred], requires_grad=False))


    I = []
    if orig_pred == 0:
        I = [0, 1]
    else:
        I = [1, 0]

    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    k_i = origin

    loop_i = 0
    while torch.eq(k_i, origin) and loop_i < maxiters:

        # Origin class
        ### PROBLEM ###
        output[I[0]].backward(retain_graph=True)

        ### This is NONE ###
        grad_orig = x.grad.data.cpu().numpy().copy()[0]

        # Target class
        if x.grad is not None:
            x.grad.zero_()
        output[I[1]].backward(retain_graph=True)

        cur_grad = x.grad.data.cpu().numpy().copy()[0]

        # set new w and new f
        w = cur_grad - grad_orig
        f = (output[I[1]] - output[I[0]]).data.cpu().numpy().copy()

        pert = abs(f)/np.linalg.norm(w.flatten())

        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i =  (pert+1e-4) * w / np.linalg.norm(w)

        if len(weights) > 0:
            r_i /= np.array(weights)

        # limit huge step
        r_i = alpha * r_i / np.linalg.norm(r_i)

        r_tot = np.float32(r_tot + r_i)


        pert_x = x_old + (1 + overshoot) * torch.from_numpy(r_tot)

        if len(bounds) > 0:
            pert_x = clip(pert_x[0], bounds[0], bounds[1]).unsqueeze(0)

        x = Variable(pert_x.to('cuda'), requires_grad=True)

        output = net.predict_proba_2(x)[0]
        k_i = output.max(0, keepdim=True)[1].to('cpu')

        loop_i += 1

    r_tot = (1+overshoot)*r_tot
    pert_x = clip(pert_x[0].to('cpu'), bounds[0], bounds[1])

    return orig_pred, k_i, pert_x.clone().detach().cpu().numpy(), loop_i
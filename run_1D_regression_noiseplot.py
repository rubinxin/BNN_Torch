#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: robin
"""
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
from tasks.math_functions import get_function
from models.gp import GPModel
from models.deepensemble import DeepEnsemble
from models.mcdrop import MCDropout
from utils.utilities import get_init_data
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# func_name = 'rastrigin-1d'
# func_name = 'ackley-1d'
# func_name = 'sinc-1d'
func_name = 'GM-1d'
# func_name = 'gramcy-1d'
f, x_bounds, _, true_fmin = get_function(func_name)
d = x_bounds.shape[0]
n_init = 60
noise_var = 0.1
seed = 5
plot_noise = True
n_units = 10
hetero = True

# model_name = 'deepensemble'
model_name_list = ['mcdropout', 'deepensemble']

#  Specify the objective function and parameters (noise variance, input dimension, initial observation
np.random.seed(seed)
hetero_f = lambda x: np.exp(x)**2

x_train_np, y_train_np = get_init_data(obj_func=f, noise_var=noise_var, n_init =n_init, bounds=x_bounds,hetero=hetero_f)

# ------ Test grid -------- #
if d == 2:
    x1, x2 = np.mgrid[-1:1:50j, -1:1:50j]
    x_val_np = np.vstack((x1.flatten(), x2.flatten())).T
    y_val_np = f(x_val_np)
    hetero_noise_var = (hetero_f(x_val_np) * np.sqrt(noise_var))**2

else:
    x_val_np = np.linspace(-1,1,100)[:,None]
    y_val_np = f(x_val_np)
    hetero_noise_var = (hetero_f(x_val_np) * np.sqrt(noise_var))**2

figure, axes = plt.subplots(1, 1, figsize=(10, 8))
axes.plot(x_train_np, y_train_np, 'ro')
axes.plot(x_val_np, y_val_np, 'k--')
plt.savefig(f'./figs/{func_name}')

models_mean = []
models_noise_var = []
models_std = []
for model_name in model_name_list:
    np.random.seed(seed)

    if model_name == 'deepensemble':
        model = DeepEnsemble(num_epochs=1000, n_units=[n_units]*3, hetero=hetero,
                                      actv='tanh', seed=seed, normalize_input=False)

    elif model_name == 'mcdropout':
        model = MCDropout(num_epochs=1000, n_units=[n_units] * 3, hetero=hetero, tau=100, length_scale=0.01,
                          T_samples= 50, dropout_p= 0.1,
                             actv='tanh', seed=seed, normalize_input=False)

    model.fit(x_train_np, y_train_np)
    m, s = model.predict(x_val_np)

    if plot_noise:
        noise_var = model.get_hetero_noise(x_val_np)
        models_noise_var.append(noise_var)

    models_mean.append(m)
    models_std.append(s)

if hetero:
    uncertainty_type = 'Heteroscedastic'
else:
    uncertainty_type = 'Homoscedastic'

if plot_noise:
    figure, axes = plt.subplots(len(models_mean), 2, figsize=(5*len(models_mean), 7))
    for i in range(len(model_name_list)):
        m = models_mean[i]
        std = models_std[i]
        noise_var = models_noise_var[i]
        #  compite the nll and rmse
        total_nll = - np.mean(norm.logpdf(y_val_np, loc=m, scale=std))
        total_mse = np.mean((y_val_np - m) ** 2)
        subtitle = f'{model_name}: nll={total_nll:.3f}, rmse={total_mse:.3f}'
        axes[i,0].plot(x_train_np, y_train_np, 'ko')
        axes[i,0].plot(x_val_np, y_val_np, 'r--')
        axes[i,0].plot(x_val_np, m, 'b')
        axes[i,0].fill_between(x_val_np.flatten(), (m - std).flatten(), (m + std).flatten(), color='blue', alpha=0.40)
        axes[i,1].plot(x_val_np, hetero_noise_var, 'r--')
        axes[i,1].plot(x_val_np, noise_var, 'b-')
        axes[i,0].set_title(subtitle)

    fig_name = f'{func_name}_{model_name}_nu{n_units}_hetero{hetero}{seed}noise.pdf'

else:
    figure, axes = plt.subplots(len(models_mean), 1, figsize=(5 * len(models_mean), 7))
    for i in range(len(model_name_list)):
        m = models_mean[i]
        std = models_std[i]
        #  compite the nll and rmse
        total_nll = - np.mean(norm.logpdf(y_val_np, loc=m, scale=std))
        total_mse = np.mean((y_val_np - m) ** 2)
        subtitle = f'{model_name}: nll={total_nll:.3f}, rmse={total_mse:.3f}'
        axes[i].plot(x_train_np, y_train_np, 'ko')
        axes[i].plot(x_val_np, y_val_np, 'r--')
        axes[i].plot(x_val_np, m, 'b')
        axes[i].fill_between(x_val_np.flatten(), (m - std).flatten(), (m + std).flatten(), color='blue', alpha=0.40)
        axes[i].set_title(subtitle)

    fig_name = f'{func_name}_{model_name}_nu{n_units}_hetero{hetero}{seed}.pdf'

figure.suptitle(f'{uncertainty_type} Uncertainty')  # or plt.suptitle('Main title')

# ------ Save figures -------- #
saving_folder = './figs/'
if not os.path.exists(saving_folder):
    os.makedirs(saving_folder)

model_name_str = ''.join(model_name_list)
saving_path = os.path.join(saving_folder,fig_name)
figure.savefig(saving_path)


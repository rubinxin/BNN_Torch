import time
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from models.base import BaseModel
from models.networks import Net, negative_log_likelihood
from utils.utilities import iterate_minibatches, zero_mean_unit_var_normalization, zero_mean_unit_var_denormalization


class DeepEnsemble(BaseModel):

    def __init__(self, batch_size=10, num_epochs=1000, ensemble_size=5,
                 learning_rate=0.01, n_units = [50, 50, 50], dropout_p=0.05,
                 wd=1e-6, normalize_input=True, normalize_output=True,
                 seed=None, gpu=True, actv='tanh',
                 hetero=True):

        # fix seeds for reproducibility
        self.seed = seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        self.rng = np.random.RandomState(seed)

        # Use GPU or CPU
        self.gpu = gpu
        self.device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")

        # specify the model
        self.actv = actv
        self.n_units = n_units
        self.ensemble_size = ensemble_size
        self.hetero = hetero
        self.dropout_p = dropout_p

        # specify training hyperparamters
        self.normalize_input = normalize_input
        self.normalize_output = normalize_output
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.init_learning_rate = learning_rate
        self.wd = wd


    def fit(self, X_train, y_train):
        """
        :param X_train: training input data (N x d)
        :param y_train: training output data (N x 1)
        :return:
        """

        # Normalize inputs
        if self.normalize_input:
            self.X_train, self.X_mean, self.X_std = zero_mean_unit_var_normalization(X_train)
        else:
            self.X_train = X_train

        # Normalize ouputs
        if self.normalize_output:
            self.y_train, self.y_mean, self.y_std = zero_mean_unit_var_normalization(y_train)
        else:
            self.y_train = y_train

        # Check if we have enough points to create a minibatch otherwise use all data points
        if self.X_train.shape[0] <= self.batch_size:
            batch_size = self.X.shape[0]
        else:
            batch_size = self.batch_size

        # Create the neural network
        input_feature_dim = X_train.shape[1]

        ensemble_networks = []
        ensemble_losses = {'train_losses': [], 'valid_losses': []}
        for i in range(self.ensemble_size):
            # create one base-learner
            network = Net(n_inputs=input_feature_dim, dropout_p=self.dropout_p,
                          hetero=self.hetero,  actv=self.actv,
                          n_units=self.n_units)
            network = network.to(self.device)
            optimizer = optim.Adam(network.parameters(),
                                   lr=self.init_learning_rate, weight_decay=self.wd)

            # Start training
            train_curve = []
            network.train()

            for epoch in range(self.num_epochs):

                train_loss = 0
                train_batches = 0

                for batch in iterate_minibatches(self.X_train, self.y_train,
                                                batch_size, self.rng,
                                                 shuffle=True):

                    inputs = Variable(torch.Tensor(batch[0]))
                    targets = Variable(torch.Tensor(batch[1]))
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    optimizer.zero_grad()
                    output = network(inputs)
                    loss = negative_log_likelihood(output, targets)

                    loss.backward()
                    optimizer.step()

                    train_loss += loss.cpu().data.numpy()
                    train_batches += 1

                train_curve.append(train_loss / train_batches)

            ensemble_losses['train_losses'].append(train_curve)
            ensemble_networks.append(network)

        self.ensemble_model = ensemble_networks

        return ensemble_losses

    def predict(self, X_test):

        # Unnormalise inputs
        if self.normalize_input:
            X_test, _, _ = zero_mean_unit_var_normalization(X_test, self.X_mean, self.X_std)
        else:
            X_test = X_test

        # Perform Ensemble Prediction
        ensemble_model = self.ensemble_model
        baselearner_mean_predicts = []
        baselearner_sec_moments_predicts = []
        X_test_tensor = Variable(torch.Tensor(X_test))
        X_test_tensor = X_test_tensor.to(self.device)

        for i in range(self.ensemble_size):
            model_i = ensemble_model[i]
            model_i.eval()

            y_pred_hat = model_i(X_test_tensor).data.numpy()

            mean_hat = y_pred_hat[:,0]
            log_var_hat = y_pred_hat[:,1]
            baselearner_mean_predicts.append(mean_hat)
            baselearner_sec_moments_predicts.append(np.exp(log_var_hat) + mean_hat**2)

        ensemble_mean = np.mean(baselearner_mean_predicts,0)
        ensemble_var = np.mean(baselearner_sec_moments_predicts, 0) - ensemble_mean ** 2
        ensemble_var = np.clip(ensemble_var, np.finfo(ensemble_var.dtype).eps, np.inf)

        # Unnormalise output
        if self.normalize_output:
            pred_mean = zero_mean_unit_var_denormalization(ensemble_mean, self.y_mean, self.y_std)
            pred_var = ensemble_var * self.y_std ** 2

        pred_std = np.sqrt(pred_var)

        return pred_mean[:,None], pred_std[:,None]

    def get_hetero_noise(self, X_test):

        if self.hetero:
            # Unnormalise inputs
            if self.normalize_input:
                X_test, _, _ = zero_mean_unit_var_normalization(X_test, self.X_mean, self.X_std)
            else:
                X_test = X_test

            # Perform Prediction
            X_test_tensor = Variable(torch.Tensor(X_test))
            X_test_tensor = X_test_tensor.to(self.device)
            baselearner_noise_var_preds = []
            for i in range(self.ensemble_size):
                model_i = self.ensemble_model[i]
                model_i.eval()
                baselearner_noise_var_preds.append(np.exp(model_i(X_test_tensor).data.numpy()[:, 1]))
            ensemble_noise_var = np.mean(baselearner_noise_var_preds, 0)
            noise_var = ensemble_noise_var * self.y_std ** 2

            return noise_var

        else:
            assert False
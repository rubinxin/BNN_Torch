import time
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from models.base import BaseModel
from models.networks import negative_log_likelihood, Net
from utils.utilities import iterate_minibatches, zero_mean_unit_var_normalization, zero_mean_unit_var_denormalization

class MCNet(Net):

    def forward(self, x):

        x = self.activation(self.fc1(x))
        x = self.dropout(x)

        x = self.activation(self.fc2(x))
        x = self.dropout(x)

        x = self.activation(self.fc3(x))

        x = self.dropout(x)

        if self.hetero:
            mean = self.out1(x)
            log_var = self.out2(x)
            return torch.cat((mean, log_var), dim=1)
        else:
            x = self.out1(x)
            return x

class MCDropout(BaseModel):

    def __init__(self, batch_size=10, num_epochs=1000,
                 tau=1.0, T_samples=100, length_scale = 0.01, dropout_p=0.05,
                 learning_rate=0.01, n_units = [50, 50, 50],
                 normalize_input=True, normalize_output=True,
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
        self.hetero = hetero
        self.dropout_p = dropout_p
        self.T_samples = T_samples
        self.tau = tau
        self.length_scale = length_scale

        # specify training hyperparamters
        self.normalize_input = normalize_input
        self.normalize_output = normalize_output
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.init_learning_rate = learning_rate

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
        N = X_train.shape[0]

        self.wd = self.length_scale**2*(1 - self.dropout_p) / (2.*N*self.tau)

        # create one base-learner
        network = MCNet(n_inputs=input_feature_dim, dropout_p=self.dropout_p,
                      hetero=self.hetero,  actv=self.actv, n_units=self.n_units)
        network = network.to(self.device)
        optimizer = optim.Adam(network.parameters(),
                               lr=self.init_learning_rate, weight_decay=self.wd)

        # Start training
        model_losses = {'train_losses': [], 'valid_losses': []}
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
                if self.hetero:
                    loss = negative_log_likelihood(output, targets)
                else:
                    # loss = negative_log_likelihood(output, targets)
                    loss = nn.functional.mse_loss(output, targets)

                loss.backward()
                optimizer.step()

                train_loss += loss.cpu().data.numpy()
                train_batches += 1

            train_curve.append(train_loss / train_batches)

        model_losses['train_losses'].append(train_curve)

        self.model = network

        return model_losses

    def predict(self, X_test):

        # Unnormalise inputs
        if self.normalize_input:
            X_test, _, _ = zero_mean_unit_var_normalization(X_test, self.X_mean, self.X_std)
        else:
            X_test = X_test

        # Perform Prediction
        model = self.model
        model.train()
        X_test_tensor = Variable(torch.Tensor(X_test))
        X_test_tensor = X_test_tensor.to(self.device)

        if self.hetero:
            MC_samples_mean_preds = []
            MC_samples_sec_moments_preds = []

            for i in range(self.T_samples):

                y_pred_hat = model(X_test_tensor).data.numpy()

                mean_hat = y_pred_hat[:, 0]
                log_var_hat = y_pred_hat[:, 1]
                MC_samples_mean_preds.append(mean_hat)
                MC_samples_sec_moments_preds.append(np.exp(log_var_hat) + mean_hat ** 2)

            mcdropout_mean = np.mean(MC_samples_mean_preds, 0)
            mcdropout_mean_var = np.mean(MC_samples_sec_moments_preds, 0) - mcdropout_mean ** 2
        else:
            mc_samples_preds = [model(X_test_tensor).data.numpy() for _ in range(self.T_samples)]
            mcdropout_mean = np.mean(mc_samples_preds, 0)
            mcdropout_mean_var = np.var(mc_samples_preds, 0) + 1./self.tau

        mcdropout_mean_var = np.clip(mcdropout_mean_var, np.finfo(mcdropout_mean_var.dtype).eps, np.inf)

        # Unnormalise output
        if self.normalize_output:
            pred_mean = zero_mean_unit_var_denormalization(mcdropout_mean, self.y_mean, self.y_std)
            pred_var = mcdropout_mean_var * self.y_std ** 2

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
            model = self.model
            model.train()
            X_test_tensor = Variable(torch.Tensor(X_test))
            X_test_tensor = X_test_tensor.to(self.device)
            mc_samples_noise_var_preds = [np.exp(model(X_test_tensor).data.numpy()[:, 1]) for _ in range(self.T_samples)]
            mcdropout_heteroscedastic_noise = np.mean(mc_samples_noise_var_preds, 0)

            noise_var = mcdropout_heteroscedastic_noise * self.y_std ** 2

            return noise_var

        else:
            assert False


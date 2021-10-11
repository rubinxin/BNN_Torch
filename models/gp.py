from .base import BaseModel
import numpy as np
import GPy

class GPModel(BaseModel):

    analytical_gradient_prediction = True

    def __init__(self, kernel=None, noise_var=None, exact_feval=False, optimizer='bfgs',
                 max_iters=1000, optimize_restarts=5,sparse = False, num_inducing = 10,
                 verbose=False, ARD=False, seed=42):
        self.kernel = kernel
        self.noise_var = noise_var
        self.exact_feval = exact_feval
        self.optimize_restarts = optimize_restarts
        self.optimizer = optimizer
        self.max_iters = max_iters
        self.verbose = verbose
        self.sparse = sparse
        self.num_inducing = num_inducing
        self.model = None
        self.ARD = ARD
        self.normalize_Y = True
        self.seed = seed
        np.random.rand(self.seed)

    def fit(self, X, Y):
        """
        Creates the model given some input data X and Y.
        """
        # --- define kernel
        self.input_dim = X.shape[1]
        if self.kernel is None:
            kern = GPy.kern.Matern52(self.input_dim, variance=1., ARD=self.ARD) #+ GPy.kern.Bias(self.input_dim)
        else:
            kern = self.kernel
            self.kernel = None

        # --- define model
        noise_var = Y.var()*0.01 if self.noise_var is None else self.noise_var

        if not self.sparse:
            self.model = GPy.models.GPRegression(X, Y, kernel=kern, noise_var=noise_var)
        else:
            self.model = GPy.models.SparseGPRegression(X, Y, kernel=kern, num_inducing=self.num_inducing)

        # --- restrict variance if exact evaluations of the objective
        if self.exact_feval:
            self.model.Gaussian_noise.constrain_fixed(1e-6, warning=False)
        else:
            # --- We make sure we do not get ridiculously small residual noise variance
            self.model.Gaussian_noise.constrain_bounded(1e-9, 1e6, warning=False) #constrain_positive(warning=False)

    def _update_model(self, X_all, Y_all_raw, itr=0):
        """
        Updates the model with new observations.
        """
        if self.normalize_Y:
            self.Y_mean = Y_all_raw.mean()
            self.Y_std = Y_all_raw.std()
            Y_all = (Y_all_raw - Y_all_raw.mean())/(Y_all_raw.std())
        else:
            Y_all = Y_all_raw

        if self.model is None:
            self._create_model(X_all, Y_all)
        else:
            self.model.set_XY(X_all, Y_all)

        # WARNING: Even if self.max_iters=0, the hyperparameters are bit modified...
        if self.max_iters > 0:
            # --- update the model maximizing the marginal likelihood.
            if self.optimize_restarts==1:
                self.model.optimize(optimizer=self.optimizer, max_iters = self.max_iters, messages=False, ipython_notebook=False)
            else:
                self.model.optimize_restarts(num_restarts=self.optimize_restarts, optimizer=self.optimizer, max_iters = self.max_iters, verbose=self.verbose)

    def predict(self, X):
        """
        Predictions with the model. Returns posterior means and standard deviations at X. Note that this is different in GPy where the variances are given.
        """
        if X.ndim==1: X = X[None,:]
        m, v = self.model.predict(X)
        v = np.clip(v, 1e-10, np.inf)

        if self.normalize_Y:
            m = (m*self.Y_std + self.Y_mean)
            v = v * self.Y_std**2
        else:
            m = m

        return m, np.sqrt(v)

    def predict_full(self, X):
        """
        Predictions with the model using the full covariance matrix
        """
        mu, cov = self.model.predict(X, full_cov=True)
        return mu, cov

    def get_fmin(self):
        """
        Returns the location where the posterior mean is takes its minimal value.
        """
        return self.model.predict(self.model.X)[0].min()

    def predict_withGradients(self, X):
        """
        Returns the mean, standard deviation, mean gradient and standard deviation gradient at X.
        """
        if X.ndim==1: X = X[None,:]
        m, v = self.model.predict(X)
        v = np.clip(v, 1e-10, np.inf)
        dmdx, dvdx = self.model.predictive_gradients(X)
        dmdx = dmdx[:,:,0]
        dsdx = dvdx / (2*np.sqrt(v))

        return m, np.sqrt(v), dmdx, dsdx

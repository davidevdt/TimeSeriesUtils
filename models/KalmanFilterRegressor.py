import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from pykalman import KalmanFilter

class KalmanFilterRegressor(BaseEstimator, RegressorMixin):
    """
    A regression model that uses a Kalman Filter for state estimation, which can be used
    to track the latent state of a time-varying system.

    Parameters
    ----------
    add_intercept : bool, optional (default=True)
        Whether to add an intercept (bias) term to the features.
    
    initial_state_mean : array-like, shape (n_dim_state,), optional (default=None)
        The initial guess for the state mean.
    
    initial_state_covariance : array-like, shape (n_dim_state, n_dim_state), optional (default=None)
        The initial guess for the state covariance.
    
    transition_matrices : array-like, shape (n_dim_state, n_dim_state), optional (default=None)
        The state transition matrix.
    
    transition_covariance : array-like, shape (n_dim_state, n_dim_state), optional (default=None)
        The covariance of the state transition.
    
    observation_covariance : array-like, shape (n_dim_obs, n_dim_obs), optional (default=None)
        The covariance of the observation noise.
    
    transition_offsets : array-like, shape (n_dim_state,), optional (default=None)
        The offset added to the state transition.
    
    observation_offsets : array-like, shape (n_dim_obs,), optional (default=None)
        The offset added to the observation.
    
    n_iter : int, optional (default=5)
        The number of iterations to perform in the EM algorithm, if `use_em=True`.
    
    em_vars : list of str, optional (default=['transition_covariance', 'observation_covariance', 'initial_state_mean', 'initial_state_covariance'])
        The variables that should be learned using the EM algorithm.
    
    use_em : bool, optional (default=False)
        Whether to use the EM algorithm to learn the parameters.
    
    use_smooth : bool, optional (default=True)
        Whether to use the Kalman Smoothing algorithm (True) or Kalman Filtering algorithm (False).

    Attributes
    ----------
    kf : KalmanFilter object
        The KalmanFilter object after fitting.
    
    state_means : array-like
        The mean of the states after fitting.
    
    state_covariances : array-like
        The covariance of the states after fitting.

    Methods
    -------
    fit(X, y)
        Fit the Kalman Filter to the data.
    
    predict(X, update_state=False)
        Predict the output using the fitted Kalman Filter.
    
    get_params(deep=True)
        Get parameters for this estimator.
    
    set_params(**parameters)
        Set the parameters of this estimator.
    """

    def __init__(self, add_intercept=True, initial_state_mean=None,
                 initial_state_covariance=None, transition_matrices=None,
                 transition_covariance=None,
                 observation_covariance=None, transition_offsets=None,
                 observation_offsets=None, n_iter=5, em_vars=None,
                 use_em=False, use_smooth=True):
        self.add_intercept = add_intercept
        self.initial_state_mean = initial_state_mean
        self.initial_state_covariance = initial_state_covariance
        self.transition_matrices = transition_matrices
        self.transition_covariance = transition_covariance
        self.observation_covariance = observation_covariance
        self.transition_offsets = transition_offsets
        self.observation_offsets = observation_offsets
        self.n_iter = n_iter
        self.em_vars = em_vars if em_vars else ['transition_covariance', 'observation_covariance', 'initial_state_mean', 'initial_state_covariance']
        self.use_em = use_em
        self.use_smooth = use_smooth

        self.kf = None
        self.state_means = None
        self.state_covariances = None

    def _convert_to_numpy(self, X, y=None):
        """Convert X and y to numpy arrays if they are pandas DataFrames."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        if y is not None:
            if isinstance(y, pd.Series):
                y = y.values
            return X, y
        return X

    def _add_intercept(self, X):
        """Add a column of ones to X if add_intercept is True."""
        if self.add_intercept:
            return np.hstack([np.ones((X.shape[0], 1)), X])
        return X

    def fit(self, X, y):
        """Fit the Kalman Filter to the data."""
        X, y = self._convert_to_numpy(X, y)
        X = self._add_intercept(X)
        n_features_total = X.shape[1]

        # Initialize parameters if not provided
        if self.initial_state_mean is None:
            self.initial_state_mean = np.zeros(n_features_total)
        if self.initial_state_covariance is None:
            self.initial_state_covariance = np.eye(n_features_total) * 1000
        if self.transition_matrices is None:
            self.transition_matrices = np.eye(n_features_total)
        if self.transition_covariance is None:
            self.transition_covariance = np.eye(n_features_total) * 0.01
        if self.observation_covariance is None:
            self.observation_covariance = np.eye(1) * 1.0

        # Create KalmanFilter object
        self.kf = KalmanFilter(
            n_dim_state=n_features_total,
            n_dim_obs=1,
            initial_state_mean=self.initial_state_mean,
            initial_state_covariance=self.initial_state_covariance,
            transition_matrices=self.transition_matrices,
            observation_matrices=X[:, np.newaxis],
            transition_covariance=self.transition_covariance,
            observation_covariance=self.observation_covariance,
            transition_offsets=self.transition_offsets,
            observation_offsets=self.observation_offsets,
            em_vars=self.em_vars
        )

        # Use EM algorithm if specified
        if self.use_em:
            self.kf = self.kf.em(y.reshape(-1, 1), n_iter=self.n_iter)

        # Fit the filter
        if self.use_smooth:
            self.state_means, self.state_covariances = self.kf.smooth(y.reshape(-1, 1))
        else:
            self.state_means, self.state_covariances = self.kf.filter(y.reshape(-1, 1))

        return self

    def predict(self, X, update_state=False):
        """Predict using the Kalman Filter."""
        if self.kf is None:
            raise ValueError("Model has not been fitted yet.")

        X = self._convert_to_numpy(X)
        X = self._add_intercept(X)

        if update_state:
            # Update state with each new observation
            for x in X:
                self.state_means, self.state_covariances = self.kf.filter_update(
                    filtered_state_mean=self.state_means[-1],
                    filtered_state_covariance=self.state_covariances[-1],
                    observation=x
                )

        predictions = X @ self.state_means[-1]
        return predictions

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            "add_intercept": self.add_intercept,
            "initial_state_mean": self.initial_state_mean,
            "initial_state_covariance": self.initial_state_covariance,
            "transition_matrices": self.transition_matrices,
            "observation_matrices": self.observation_matrices,
            "transition_covariance": self.transition_covariance,
            "observation_covariance": self.observation_covariance,
            "transition_offsets": self.transition_offsets,
            "observation_offsets": self.observation_offsets,
            "n_iter": self.n_iter,
            "em_vars": self.em_vars,
            "use_em": self.use_em,
            "use_smooth": self.use_smooth
        }

    def set_params(self, **parameters):
        """Set the parameters of this estimator."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

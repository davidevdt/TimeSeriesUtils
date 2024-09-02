# TimeSeriesUtils

`TimeSeriesUtils` is a Python package designed to simplify time series analysis and modeling. It provides utilities for feature selection, model comparison, and forecasting, making it easier to build and evaluate time series models.

## Features

- **Feature Selection**: Automatically select relevant features for time series forecasting using various models and parameter grids.
- **Model Comparison**: Compare multiple models and hyperparameter settings to find the best model for your time series data.
- **Forecasting**: Generate forecasts using iterative models with support for lags and custom transformers.

## Installation

You can install `TimeSeriesUtils` via pip. To do this, clone the repository and install the package in your environment:

```bash
git clone https://github.com/yourusername/TimeSeriesUtils.git
cd TimeSeriesUtils
pip install .
```

Alternatively, you can install the dependencies directly from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Usage

### Feature Selection

```python
from timeseriesutils.feature_selection import FeatureSelection
from sklearn.linear_model import Lasso
from sklearn.model_selection import TimeSeriesSplit

# Define your models and parameter grids
models = [Lasso()]
param_grids = {
    'Lasso': {
        'alpha': [0.1, 1.0, 10.0]
    }
}
cv_splitter = TimeSeriesSplit(n_splits=5)

# Initialize the FeatureSelection object
feature_selector = FeatureSelection(models=models, params_grids=param_grids, cv_splitter=cv_splitter)

# Fit the feature selector
feature_selector.fit(X_train, y_train)

# Get the selected features
selected_features = feature_selector.selected_features
```

### Model Comparison

```python
from timeseriesutils.model_comparison import ModelComparison
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

# Define your models and parameter grids
models = [RandomForestRegressor()]
param_grids = {
    'RandomForestRegressor': {
        'n_estimators': [100, 200],
        'max_depth': [10, 20]
    }
}
cv_splitter = TimeSeriesSplit(n_splits=5)

# Initialize the ModelComparison object
model_comparator = ModelComparison(models=models, param_grids=param_grids, cv_splitter=cv_splitter)

# Fit the model comparator
model_comparator.fit(X_train, y_train)

# Get the best model and its parameters
best_model = model_comparator.get_best_model()
best_params = model_comparator.best_params
```

### Forecasting

```python
from timeseriesutils.iterative_forecaster import IterativeForecaster
from sklearn.linear_model import Lasso

# Initialize the IterativeForecaster
forecaster = IterativeForecaster(model=Lasso(), pre_trained=False)

# Fit the forecaster
forecaster.fit(X_train, y_train)

# Forecast future values
forecasts = forecaster.forecast(X_test=X_test, horizon=10)
```

## Contributing

We welcome contributions to `TimeSeriesUtils`. Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## License

`TimeSeriesUtils` is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

# TimeSeriesUtils

`timeseriesutils` is a Python package designed to provide utilities for time series analysis and forecasting. It includes tools for time series feature extraction, forecasting with various models, and handling time series data with ease.

## Features

- **Time Series Utilities**: Functions to handle and manipulate time series data.
- **Iterative Forecaster**: A custom forecasting model that iteratively predicts future values using a provided regression model.
- **Moving Window Splitter**: A splitter for time series cross-validation using a moving window approach.
- others, such as a Kalman Filter regressor module compatible with the scikit-learn interface 

## Installation
You can install `timeseriesutils` using pip. Run the following command:

```bash
git clone https://github.com/yourusername/timeseriesutils.git
cd timeseriesutils
pip install .
```

## Dependencies
The package depends on the following libraries:

- `numpy`
- `pandas`
- `scikit-learn`
- `pykalman`

These dependencies will be automatically installed when you install timeseriesutils via pip.

## Usage 

### Time Series Utilities
The TimeSeriesUtils class provides various utilities for time series data, including:

- Feature Extraction: Extract features like hour, minute, and second from datetime columns.
- Standardizing Frequencies: Align time series data to a consistent frequency.
- Time Zone Conversion: Convert between UTC and other time zones.

### Kalman Filter Regressor
To use the KalmanFilterRegressor:

```python
from timeseriesutils.kalman_filters import KalmanFilterRegressor

# Create and fit the model
model = KalmanFilterRegressor()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
``` 

### Iterative Forecaster
To use the IterativeForecaster:

```python
from timeseriesutils.forecasters import IterativeForecaster

# Create and fit the model
forecaster = IterativeForecaster(model=your_model)
forecaster.fit(X_train, y_train)

# Forecast
forecast = forecaster.forecast(y_train=y_train, X_test=X_test, steps=10)
```

### Moving Window Splitter
To use the `MovingWindowSplitter`:

```python
from timeseriesutils.model_selection import MovingWindowSplitter

# Initialize the splitter
splitter = MovingWindowSplitter(train_size=0.5, test_size=0.2)

# Generate splits
for train_index, test_index in splitter.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
```

## Contributing
Contributions are welcome! If you have suggestions or encounter issues, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the `[LICENSE](LICENSE)` file for details.



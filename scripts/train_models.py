import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import matplotlib.pyplot as plt


class TimeSeriesModel:
    def __init__(self, data_path):
        """
        Initialize the TimeSeriesModel class by loading and preprocessing data.
        Reads CSV data, converts 'Date' to datetime, and splits into train and test sets.
        """
        self.df = pd.read_csv(data_path)
        self.df["Date"] = pd.to_datetime(
            self.df["Date"],
            format="mixed",
        )

        self.df.set_index("Date", inplace=True)
        self.df.sort_index(inplace=True)
        self.train_size = int(len(self.df) * 0.8)
        self.train, self.test = self.df[: self.train_size], self.df[self.train_size :]

    def fit_arima(self, order=(1, 1, 1)):
        """
        Train an ARIMA model with the specified order.
        Stores the trained model and saves it as a pickle file.
        """
        self.arima_model = ARIMA(self.train["Price"], order=order)
        self.arima_result = self.arima_model.fit()
        with open("../checkpoints/arima_model.pkl", "wb") as f:
            pickle.dump(self.arima_result, f)
        return self.arima_result.summary()

    def fit_garch(self, p=1, q=1):
        """
        Train a GARCH model to capture volatility clustering in price changes.
        Stores the trained model and saves it as a pickle file.
        """
        returns = self.train["Price"].pct_change().dropna()
        self.garch_model = arch_model(returns, vol="Garch", p=p, q=q)
        self.garch_result = self.garch_model.fit(disp="off")
        with open("../checkpoints/garch_model.pkl", "wb") as f:
            pickle.dump(self.garch_result, f)
        return self.garch_result.summary()

    def fit_random_forest(self, n_estimators=100):
        """
        Train a Random Forest model for time series forecasting.
        Uses sequential integer indices as features and stores the trained model.
        """
        X_train = np.arange(len(self.train)).reshape(-1, 1)
        y_train = self.train["Price"].values

        self.rf_model = RandomForestRegressor(n_estimators=n_estimators)
        self.rf_model.fit(X_train, y_train)

        with open("../checkpoints/random_forest_model.pkl", "wb") as f:
            pickle.dump(self.rf_model, f)

        return "Random Forest model trained and saved."

    def evaluate_model(self, model_type="arima"):
        """
        Evaluate the specified model using RMSE and MAE metrics.
        Generates predictions and plots actual vs predicted values.
        Loads the trained model from the checkpoint before evaluation.
        """
        if model_type == "arima":
            with open("../checkpoints/arima_model.pkl", "rb") as f:
                self.arima_result = pickle.load(f)
            predictions = self.arima_result.forecast(steps=len(self.test))
        elif model_type == "garch":
            with open("../checkpoints/garch_model.pkl", "rb") as f:
                self.garch_result = pickle.load(f)
            predictions = self.garch_result.conditional_volatility[-len(self.test) :]
        elif model_type == "random_forest":
            with open("../checkpoints/random_forest_model.pkl", "rb") as f:
                self.rf_model = pickle.load(f)
            X_test = np.arange(len(self.train), len(self.df)).reshape(-1, 1)
            predictions = self.rf_model.predict(X_test)
        else:
            raise ValueError(
                "Unsupported model type. Choose 'arima', 'garch', or 'random_forest'"
            )

        rmse = np.sqrt(mean_squared_error(self.test["Price"], predictions))
        mae = mean_absolute_error(self.test["Price"], predictions)

        # Visualization
        plt.figure(figsize=(10, 5))
        plt.plot(
            self.test.index, self.test["Price"], label="Actual Price", color="blue"
        )
        plt.plot(
            self.test.index,
            predictions,
            label=f"{model_type.upper()} Prediction",
            color="red",
        )
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.title(f"Actual vs Predicted Prices ({model_type.upper()})")
        plt.legend()
        plt.show()

        return {"RMSE": rmse, "MAE": mae}

    def predict_future(self, steps=30, model_type="arima"):
        """
        Predict future prices using the trained model.
        Supports ARIMA, GARCH, and Random Forest models.
        """
        if model_type == "arima":
            forecast = self.arima_result.forecast(steps=steps)
        elif model_type == "garch":
            forecast = self.garch_result.forecast(
                start=len(self.train), horizon=steps
            ).variance.mean(axis=1)
        elif model_type == "random_forest":
            future_indices = np.arange(len(self.df), len(self.df) + steps).reshape(
                -1, 1
            )
            forecast = self.rf_model.predict(future_indices)
        else:
            raise ValueError(
                "Unsupported model type. Choose 'arima', 'garch', or 'random_forest'"
            )

        return forecast

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
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

        # Initialize scalers
        self.scaler = StandardScaler()

    def create_features(self, data):
        """Create time series features for the models."""
        df = data.copy()
        df["year"] = df.index.year
        df["month"] = df.index.month
        df["day"] = df.index.day
        df["day_of_week"] = df.index.dayofweek

        # Add lag features
        for lag in [1, 2, 3, 5, 7, 14, 21]:
            df[f"price_lag_{lag}"] = df["Price"].shift(lag)

        # Add rolling statistics
        for window in [7, 14, 30]:
            df[f"rolling_mean_{window}"] = df["Price"].rolling(window=window).mean()
            df[f"rolling_std_{window}"] = df["Price"].rolling(window=window).std()

        # Add price momentum
        df["price_momentum"] = df["Price"].pct_change()

        return df.dropna()

    def prepare_data_for_ml(self, data):
        """Prepare features and target for machine learning models."""
        feature_columns = [col for col in data.columns if col != "Price"]
        X = data[feature_columns]
        y = data["Price"]
        return X, y

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

    def fit_xgboost(self):
        """Train an XGBoost model with optimized parameters."""
        # Prepare data with features
        train_data = self.create_features(self.train)
        X_train, y_train = self.prepare_data_for_ml(train_data)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)

        # XGBoost parameters optimized for time series
        params = {
            "objective": "reg:squarederror",
            "n_estimators": 1000,
            "max_depth": 5,
            "learning_rate": 0.01,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
        }

        self.xgb_model = xgb.XGBRegressor(**params)
        # Simplified fit method without early stopping
        self.xgb_model.fit(X_train_scaled, y_train)

        with open("../checkpoints/xgboost_model.pkl", "wb") as f:
            pickle.dump((self.xgb_model, self.scaler), f)

        return "XGBoost model trained and saved."

    def fit_lightgbm(self):
        """Train a LightGBM model with optimized parameters."""
        # Prepare data with features
        train_data = self.create_features(self.train)
        X_train, y_train = self.prepare_data_for_ml(train_data)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)

        # LightGBM parameters
        params = {
            "objective": "regression",
            "n_estimators": 1000,
            "num_leaves": 31,
            "learning_rate": 0.01,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
        }

        self.lgb_model = lgb.LGBMRegressor(**params)
        # Basic fit without additional parameters
        self.lgb_model.fit(X_train_scaled, y_train)

        with open("../checkpoints/lightgbm_model.pkl", "wb") as f:
            pickle.dump((self.lgb_model, self.scaler), f)

        return "LightGBM model trained and saved."

    def fit_stacking_ensemble(self):
        """Create and train a stacking ensemble of multiple models."""
        # Prepare data with features
        train_data = self.create_features(self.train)
        X_train, y_train = self.prepare_data_for_ml(train_data)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Initialize base models
        base_models = {
            "rf": RandomForestRegressor(n_estimators=100, random_state=42),
            "xgb": xgb.XGBRegressor(n_estimators=100, learning_rate=0.01),
            "lgb": lgb.LGBMRegressor(n_estimators=100, learning_rate=0.01),
            "gbm": GradientBoostingRegressor(n_estimators=100, learning_rate=0.01),
        }

        # Train base models using time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        base_predictions = np.zeros((len(X_train_scaled), len(base_models)))

        for i, (name, model) in enumerate(base_models.items()):
            model.fit(X_train_scaled, y_train)
            base_predictions[:, i] = model.predict(X_train_scaled)

        # Train meta-model
        self.meta_model = LinearRegression()
        self.meta_model.fit(base_predictions, y_train)

        # Save the ensemble
        self.ensemble = {
            "base_models": base_models,
            "meta_model": self.meta_model,
            "scaler": self.scaler,
        }

        with open("../checkpoints/stacking_ensemble.pkl", "wb") as f:
            pickle.dump(self.ensemble, f)

        return "Stacking ensemble trained and saved."

    def evaluate_model(self, model_type="ensemble"):
        """
        Evaluate the specified model using multiple metrics.
        Supports ARIMA, GARCH, Random Forest, XGBoost, LightGBM, and ensemble models.
        """
        if model_type in ["xgboost", "lightgbm", "ensemble"]:
            # Check if the requested model has been trained
            if model_type == "xgboost" and not hasattr(self, "xgb_model"):
                raise ValueError(
                    "XGBoost model has not been trained. Call fit_xgboost() first."
                )
            elif model_type == "lightgbm" and not hasattr(self, "lgb_model"):
                raise ValueError(
                    "LightGBM model has not been trained. Call fit_lightgbm() first."
                )
            elif model_type == "ensemble" and not hasattr(self, "ensemble"):
                raise ValueError(
                    "Ensemble model has not been trained. Call fit_stacking_ensemble() first."
                )

            test_data = self.create_features(self.test)
            X_test, y_test = self.prepare_data_for_ml(test_data)
            X_test_scaled = self.scaler.transform(X_test)

            if model_type == "ensemble":
                # Make predictions with base models
                base_predictions = np.zeros(
                    (len(X_test_scaled), len(self.ensemble["base_models"]))
                )
                for i, (name, model) in enumerate(self.ensemble["base_models"].items()):
                    base_predictions[:, i] = model.predict(X_test_scaled)

                # Make final predictions with meta-model
                predictions = self.meta_model.predict(base_predictions)
            else:
                model = self.xgb_model if model_type == "xgboost" else self.lgb_model
                predictions = model.predict(X_test_scaled)

            actual = y_test
            dates = test_data.index

        elif model_type == "arima":
            with open("../checkpoints/arima_model.pkl", "rb") as f:
                self.arima_result = pickle.load(f)
            predictions = self.arima_result.forecast(steps=len(self.test))
            actual = self.test["Price"]
            dates = self.test.index

        elif model_type == "garch":
            with open("../checkpoints/garch_model.pkl", "rb") as f:
                self.garch_result = pickle.load(f)
            predictions = self.garch_result.forecast(horizon=len(self.test)).mean[-1]
            actual = self.test["Price"]
            dates = self.test.index

        elif model_type == "random_forest":
            with open("../checkpoints/random_forest_model.pkl", "rb") as f:
                self.rf_model = pickle.load(f)
            X_test = np.arange(len(self.train), len(self.df)).reshape(-1, 1)
            predictions = self.rf_model.predict(X_test)
            actual = self.test["Price"]
            dates = self.test.index

        else:
            raise ValueError(
                "Unsupported model type. Choose from: 'arima', 'garch', 'random_forest', 'xgboost', 'lightgbm', or 'ensemble'"
            )

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(actual, predictions))
        mae = mean_absolute_error(actual, predictions)
        r2 = r2_score(actual, predictions)

        # Visualization
        plt.figure(figsize=(12, 6))
        plt.plot(dates, actual, label="Actual Price", color="blue")
        plt.plot(
            dates,
            predictions,
            label=f"{model_type.upper()} Prediction",
            color="red",
        )
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.title(f"Actual vs Predicted Prices ({model_type.upper()})")
        plt.legend()
        plt.grid(True)
        plt.show()

        return {"RMSE": rmse, "MAE": mae, "R2 Score": r2}

    def predict_future(self, steps=30, model_type="ensemble"):
        """
        Predict future prices using the specified model.
        Supports all implemented models including the ensemble.
        """
        # Create features for the last available data points
        last_data = self.df.tail(max(30, steps))
        future_dates = pd.date_range(
            start=self.df.index[-1] + pd.Timedelta(days=1), periods=steps, freq="D"
        )

        if model_type == "ensemble":
            predictions = []
            current_data = last_data.copy()

            for _ in range(steps):
                features = self.create_features(current_data)
                X, _ = self.prepare_data_for_ml(features)
                X_scaled = self.scaler.transform(X.iloc[[-1]])

                # Get predictions from base models
                base_predictions = np.zeros((1, len(self.ensemble["base_models"])))
                for i, (name, model) in enumerate(self.ensemble["base_models"].items()):
                    base_predictions[0, i] = model.predict(X_scaled)

                # Make final prediction with meta-model
                pred = self.meta_model.predict(base_predictions)[0]
                predictions.append(pred)

                # Update current_data for next prediction
                new_row = pd.DataFrame(
                    {"Price": pred}, index=[future_dates[len(predictions) - 1]]
                )
                current_data = pd.concat([current_data, new_row])

        elif model_type in ["xgboost", "lightgbm"]:
            # Similar recursive prediction for XGBoost and LightGBM
            model = self.xgb_model if model_type == "xgboost" else self.lgb_model
            predictions = []
            current_data = last_data.copy()

            for _ in range(steps):
                features = self.create_features(current_data)
                X, _ = self.prepare_data_for_ml(features)
                X_scaled = self.scaler.transform(X.iloc[[-1]])
                pred = model.predict(X_scaled)[0]
                predictions.append(pred)

                new_row = pd.DataFrame(
                    {"Price": pred}, index=[future_dates[len(predictions) - 1]]
                )
                current_data = pd.concat([current_data, new_row])
        else:
            # Handle existing models (ARIMA, GARCH, Random Forest)
            return super().predict_future(steps, model_type)

        return pd.Series(predictions, index=future_dates)

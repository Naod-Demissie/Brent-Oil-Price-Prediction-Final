import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import norm
from IPython.display import display


class TimeSeriesEDA:
    def __init__(self, df):
        self.df = df.copy()
        self.convert_dates()

    def convert_dates(self):
        """Convert Date column to datetime format and set as index."""
        self.df["Date"] = pd.to_datetime(self.df["Date"], errors="coerce")
        self.df.set_index("Date", inplace=True)
        self.df.sort_index(inplace=True)
        print("✔ Date column converted to datetime and set as index.")

    def data_summary(self):
        """Print basic dataset information."""
        print("Dataset Info:\n")
        print(self.df.info())
        print("\nShape of dataset:", self.df.shape)
        print("\nBasic Statistics:")
        display(self.df.describe())

    def check_missing_values(self):
        """Check for missing values in the dataset."""
        missing_values = self.df.isnull().sum()
        print("Missing Values per Column:")
        print(missing_values)

    def summary_statistics(self):
        """Display descriptive statistics of the Price column."""
        print("Summary Statistics for Price:")
        print(self.df["Price"].describe())

    def plot_moving_average(self, window=30):
        """Plot moving average of oil prices."""
        plt.figure(figsize=(14, 5))
        self.df["Price"].rolling(window=window).mean().plot(
            label=f"{window}-Day MA", color="red"
        )
        plt.plot(self.df["Price"], alpha=0.5, label="Actual Price")
        plt.title(f"{window}-Day Moving Average of Oil Prices")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid()
        plt.show()

    def plot_trends(self):
        """Decompose the time series into trend, seasonal, and residual components."""
        decomposition = seasonal_decompose(
            self.df["Price"], model="additive", period=365
        )
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10))
        decomposition.trend.plot(ax=ax1, title="Overall Trend", color="green")
        decomposition.seasonal.plot(ax=ax2, title="Seasonality", color="orange")
        decomposition.resid.plot(ax=ax3, title="Residuals", color="red")
        plt.tight_layout()
        plt.show()

    def plot_monthly_yearly_trends(self):
        """Plot monthly and yearly trends of oil prices."""
        self.df["Year"] = self.df.index.year
        self.df["Month"] = self.df.index.month
        monthly_avg = self.df.groupby("Month")["Price"].mean()
        yearly_avg = self.df.groupby("Year")["Price"].mean()

        fig, ax = plt.subplots(2, 1, figsize=(14, 8))
        monthly_avg.plot(ax=ax[0], title="Average Monthly Price Trend", color="blue")
        yearly_avg.plot(ax=ax[1], title="Average Yearly Price Trend", color="green")
        plt.tight_layout()
        plt.show()

    def plot_distribution(self):
        """Plot histogram and KDE of oil prices."""
        plt.figure(figsize=(14, 5))
        sns.histplot(self.df["Price"], kde=True, bins=50, color="purple")
        plt.title("Oil Price Distribution")
        plt.xlabel("Price")
        plt.ylabel("Frequency")
        plt.grid()
        plt.show()

    def detect_outliers(self):
        """Detect outliers using Z-score method."""
        z_scores = np.abs(
            (self.df["Price"] - self.df["Price"].mean()) / self.df["Price"].std()
        )
        outliers = self.df[z_scores > 3]
        print("Number of Outliers Detected:", len(outliers))
        return outliers

    def plot_acf_pacf(self, lags=50):
        """Plot ACF and PACF to analyze autocorrelation."""
        fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
        plot_acf(self.df["Price"], lags=lags, ax=ax[0])
        plot_pacf(self.df["Price"], lags=lags, ax=ax[1])
        ax[0].set_title("Autocorrelation Function (ACF)")
        ax[1].set_title("Partial Autocorrelation Function (PACF)")
        # add grid

        plt.show()

    def decompose_time_series(self):
        """Perform seasonal decomposition of time series."""
        decomposition = seasonal_decompose(
            self.df["Price"], model="additive", period=365
        )
        fig = plt.figure(figsize=(14, 8))
        decomposition.plot()
        plt.show()

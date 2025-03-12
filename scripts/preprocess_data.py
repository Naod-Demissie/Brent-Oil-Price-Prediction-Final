import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore


class DataPreprocessor:
    def __init__(self, df):
        """Initialize the DataPreprocessor with a pandas DataFrame."""
        self.df = df.copy()

    def missing_values_proportions(self):
        """Calculate the proportion of missing values in the DataFrame."""
        missing_values = self.df.isnull().sum()
        missing_values = missing_values[missing_values > 0]

        missing_proportions = (missing_values / len(self.df)) * 100
        missing_proportions = missing_proportions.round(2)

        return pd.DataFrame(
            {"Missing Values": missing_values, "Proportion (%)": missing_proportions}
        )

    def handle_outliers(self, columns, plot_box=False, replace_with="boundaries"):
        """Detect and handle outliers in specified columns of the DataFrame."""
        for col in columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            if replace_with == "boundaries":
                # Replace outliers with boundaries
                self.df[col] = self.df[col].clip(lower=lower_bound, upper=upper_bound)
            elif replace_with == "mean":
                # Replace outliers with the mean
                mean = self.df[col].mean()
                self.df[col] = np.where(
                    (self.df[col] < lower_bound) | (self.df[col] > upper_bound),
                    mean,
                    self.df[col],
                )
            else:
                raise ValueError("replace_with must be either 'boundaries' or 'mean'")

            if plot_box:
                plt.figure(figsize=(6, 0.3))
                sns.boxplot(x=self.df[col])
                plt.title(f"Box Plot for {col} replaced with {replace_with} ")
                plt.show()

        return self.df

    def detect_outliers(self, column):
        """Detects outliers in a numerical column using a boxplot and Z-score analysis."""
        # Boxplot visualization
        plt.figure(figsize=(5, 1.2))
        self.df.boxplot(column=column, vert=False)
        plt.title(f"Boxplot for {column}")
        plt.xlabel(column)
        plt.show()

        # Z-score analysis for outliers
        z_scores = zscore(self.df[column])
        outliers = (np.abs(z_scores) > 3).sum()
        print("\nColumns with Potential Outliers Z-Score Analysis:\n")
        print(f"{column:>45}: {outliers} potential outliers")

    def get_processed_data(self):
        """Return the processed DataFrame."""
        return self.df.copy()

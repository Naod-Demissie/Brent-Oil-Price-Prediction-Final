from flask import Flask, request, jsonify
import pandas as pd
import statsmodels.api as sm
from flask_cors import CORS

import logging
import os

app = Flask(__name__)
CORS(app)

log_dir = "/home/naod/Projects/tenx/W10/Brent-Oil-Price-Prediction/logs"

# Configure logging to file
logging.basicConfig(
    filename=os.path.join(log_dir, "app.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="a",
)
logger = logging.getLogger(__name__)


def load_data():
    file_path = "/home/naod/Projects/tenx/W10/Brent-Oil-Price-Prediction/data/raw/BrentOilPrices.csv"
    logger.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path, parse_dates=["Date"], dayfirst=True)
    df.set_index("Date", inplace=True)
    return df


def compute_moving_average(df, window=30):
    logger.info("Computing moving average")
    df["Moving_Avg"] = df["Price"].rolling(window=window).mean()
    return df


def decompose_series(df, period=365):
    logger.info("Performing time series decomposition")
    decomposition = sm.tsa.seasonal_decompose(
        df["Price"], model="additive", period=period
    )
    return decomposition


@app.route("/analyze", methods=["GET"])
def analyze():
    try:
        logger.info("Starting analysis")
        df = load_data()
        df = compute_moving_average(df)
        decomposition = decompose_series(df)

        result = {
            "moving_average": df[["Price", "Moving_Avg"]]
            .dropna()
            .reset_index()
            .to_dict(orient="records"),
            "trend": {
                str(k): v for k, v in decomposition.trend.dropna().to_dict().items()
            },
            "seasonality": {
                str(k): v for k, v in decomposition.seasonal.dropna().to_dict().items()
            },
            "residual": {
                str(k): v for k, v in decomposition.resid.dropna().to_dict().items()
            },
        }
        logger.info("Analysis completed successfully")
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)

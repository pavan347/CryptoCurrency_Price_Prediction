from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime

# Set Matplotlib to non-interactive backend
matplotlib.use('Agg')

# Flask app initialization
app = Flask(__name__)

# Load pre-trained model
model = load_model("model.keras")

# Helper function to convert Matplotlib plots to HTML
def plot_to_html(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    buf.close()
    return f"data:image/png;base64,{data}"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        stock = request.form.get("stock")
        no_of_days = int(request.form.get("no_of_days"))
        return redirect(url_for("predict", stock=stock, no_of_days=no_of_days))
    return render_template("index.html")

@app.route("/predict")
def predict():
    stock = request.args.get("stock", "BTC-USD")
    no_of_days = int(request.args.get("no_of_days", 10))

    # Fetch stock data
    end = datetime.now()
    start = datetime(end.year - 10, end.month, end.day)
    stock_data = yf.download(stock, start, end)

    if stock_data.empty:
        return render_template("result.html", error="Invalid stock ticker or no data available.")

    # Data preparation
    splitting_len = int(len(stock_data) * 0.9)
    x_test = stock_data[['Close']][splitting_len:]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(x_test)

    # Create input sequences
    x_data, y_data = [], []
    for i in range(100, len(scaled_data)):
        x_data.append(scaled_data[i - 100:i])
        y_data.append(scaled_data[i])

    x_data, y_data = np.array(x_data), np.array(y_data)

    # Generate predictions
    predictions = model.predict(x_data)
    inv_predictions = scaler.inverse_transform(predictions)
    inv_y_test = scaler.inverse_transform(y_data)

    # Prepare data for plotting
    plotting_data = pd.DataFrame({
        'Original Test Data': inv_y_test.flatten(),
        'Predicted Test Data': inv_predictions.flatten()
    }, index=x_test.index[100:])

    # Generate plots
    original_plot = create_plot(stock_data, "Closing Prices Over Time", "Close Price")
    predicted_plot = create_comparison_plot(plotting_data, "Original vs Predicted Closing Prices")
    future_plot, future_predictions = create_future_predictions_plot(stock_data, scaler, no_of_days)

    print(future_predictions)

    return render_template(
        "result.html",
        stock=stock,
        original_plot=original_plot,
        predicted_plot=predicted_plot,
        future_plot=future_plot,
        future_predictions=future_predictions,
        enumerate=enumerate,
    )

def create_plot(data, title, ylabel):
    fig = plt.figure(figsize=(15, 6))
    plt.plot(data['Close'], label='Close Price', color='blue')
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.legend()
    return plot_to_html(fig)

def create_comparison_plot(data, title):
    fig = plt.figure(figsize=(15, 6))
    plt.plot(data['Original Test Data'], label="Original Test Data")
    plt.plot(data['Predicted Test Data'], label="Predicted Test Data", linestyle="--")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    return plot_to_html(fig)

def create_future_predictions_plot(data, scaler, days):
    last_100 = data[['Close']].tail(100)
    last_100_scaled = scaler.transform(last_100).reshape(1, -1, 1)
    
    future_predictions = []
    for _ in range(days):
        next_day = model.predict(last_100_scaled)
        future_predictions.append(scaler.inverse_transform(next_day).flatten()[0])
        last_100_scaled = np.append(last_100_scaled[:, 1:, :], next_day.reshape(1, 1, -1), axis=1)
    
    fig = plt.figure(figsize=(15, 6))
    plt.plot(range(1, days + 1), future_predictions, marker='o', label="Predicted Future Prices", color="purple")
    plt.title("Future Close Price Predictions")
    plt.xlabel("Days Ahead")
    plt.ylabel("Predicted Close Price")
    plt.grid(alpha=0.3)
    plt.legend()
    return plot_to_html(fig), future_predictions

if __name__ == "__main__":
    app.run(debug=True)

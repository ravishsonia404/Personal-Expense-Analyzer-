from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import os

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        file = request.files["file"]
        data = pd.read_csv(file, parse_dates=['Date'])

        # Monthly calculation
        data['Month'] = data['Date'].dt.to_period('M')
        monthly = data.groupby('Month')['Amount'].sum()

        # Prepare ML data
        monthly_df = monthly.reset_index()
        monthly_df['MonthNum'] = np.arange(len(monthly_df))

        X = monthly_df[['MonthNum']]
        y = monthly_df['Amount']

        # Train model
        model = LinearRegression()
        model.fit(X, y)

        # Predict next month
        next_month = np.array([[len(monthly_df)]])
        prediction = model.predict(next_month)[0]

        # Plot graph
        plt.figure()
        plt.plot(monthly_df['MonthNum'], monthly_df['Amount'], marker='o')
        plt.title("Monthly Spending")
        plt.xlabel("Month Number")
        plt.ylabel("Amount")

        # Save chart
        chart_path = os.path.join("static", "chart.png")
        plt.savefig(chart_path)
        plt.close()

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
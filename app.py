from flask import Flask, render_template, request
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import pandas as pd
import numpy as np
import os

# Fix for matplotlib (VERY IMPORTANT for server)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

# Ensure folders exist
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("static", exist_ok=True)

# ✅ PDF Function (ADD HERE)
def generate_pdf(prediction, total, category):
    pdf_path = "static/report.pdf"

    doc = SimpleDocTemplate(pdf_path)
    styles = getSampleStyleSheet()

    content = []

    content.append(Paragraph("SmartFinance AI Report", styles['Title']))
    content.append(Spacer(1, 20))

    content.append(Paragraph(f"Predicted Expense: ₹{prediction}", styles['Normal']))
    content.append(Paragraph(f"Total Spending: ₹{total}", styles['Normal']))
    content.append(Paragraph(f"Top Category: {category}", styles['Normal']))

    doc.build(content)

    return pdf_path

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    total_spent = None
    max_category = None
    pdf_file = None

    if request.method == "POST":
        try:
            file = request.files.get("file")

            if not file or file.filename == "":
                error = "❌ No file uploaded"
                return render_template("index.html", error=error)

            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            # Read CSV
            data = pd.read_csv(filepath)

            # Check required columns
            required_cols = ["Date", "Category", "Amount"]
            for col in required_cols:
                if col not in data.columns:
                    error = f"❌ Missing column: {col}"
                    return render_template("index.html", error=error)

            # Convert Date
            # Clean data
            data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
            data['Amount'] = pd.to_numeric(data['Amount'], errors='coerce')

            data = data.dropna(subset=['Date', 'Amount'])

            # Monthly grouping
            data['Month'] = data['Date'].dt.to_period('M')
            monthly = data.groupby('Month')['Amount'].sum()

            monthly_df = monthly.reset_index()
            monthly_df['MonthNum'] = np.arange(len(monthly_df))

            # ML Model
            X = monthly_df[['MonthNum']]
            y = monthly_df['Amount']

            model = RandomForestRegressor()
            model.fit(X, y)

            next_month = np.array([[len(monthly_df)]])
            prediction = round(model.predict(next_month)[0], 2)

            # Insights
            total_spent = int(data['Amount'].sum())
            max_category = data.groupby("Category")["Amount"].sum().idxmax()

            # 📊 Bar Chart
            plt.figure()
            monthly.plot(kind='bar')
            plt.title("Monthly Expenses")
            plt.savefig("static/chart.png")
            plt.close()

            # 🥧 Pie Chart
            plt.figure()
            data.groupby("Category")["Amount"].sum().plot(kind='pie', autopct='%1.1f%%')
            plt.title("Category Distribution")
            plt.ylabel("")
            plt.savefig("static/pie.png")
            plt.close()

            # ✅ Generate PDF (ADD HERE)
            pdf_file = generate_pdf(prediction, total_spent, max_category)

        except Exception as e:
            error = f"⚠️ Error: {str(e)}"

    return render_template("index.html",
                           prediction=prediction,
                           total=total_spent,
                           category=max_category,
                           pdf_file=pdf_file)

print("PDF GENERATED")

# Render Deployment Fix
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

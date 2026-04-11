from flask import Flask, render_template, request
import os
import pandas as pd
import numpy as np

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# Charts
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ML
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

# ---------------- FOLDERS ----------------
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("static", exist_ok=True)

# ---------------- PDF FUNCTION ----------------
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

# ---------------- MAIN ROUTE ----------------
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    total_spent = None
    max_category = None
    pdf_file = None
    message = None

    if request.method == "POST":

        # 👉 CASE 2: FILE UPLOAD (AI PROJECT)
        if "file" in request.files:
            file = request.files.get("file")

            if file and file.filename != "":
                filepath = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(filepath)

                data = pd.read_csv(filepath)

                data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
                data['Amount'] = pd.to_numeric(data['Amount'], errors='coerce')
                data = data.dropna(subset=['Date', 'Amount'])

                data['Month'] = data['Date'].dt.to_period('M')
                monthly = data.groupby('Month')['Amount'].sum()

                monthly_df = monthly.reset_index()
                monthly_df['MonthNum'] = np.arange(len(monthly_df))

                X = monthly_df[['MonthNum']]
                y = monthly_df['Amount']

                model = RandomForestRegressor()
                model.fit(X, y)

                next_month = np.array([[len(monthly_df)]])
                prediction = round(model.predict(next_month)[0], 2)

                total_spent = int(data['Amount'].sum())
                max_category = data.groupby("Category")["Amount"].sum().idxmax()

                # Charts
                plt.figure()
                monthly.plot(kind='bar')
                plt.savefig("static/chart.png")
                plt.close()

                plt.figure()
                data.groupby("Category")["Amount"].sum().plot(kind='pie', autopct='%1.1f%%')
                plt.savefig("static/pie.png")
                plt.close()

                # PDF
                pdf_file = generate_pdf(prediction, total_spent, max_category)

    return render_template("index.html",
                           prediction=prediction,
                           total=total_spent,
                           category=max_category,
                           pdf_file=pdf_file,
                           message=message)

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)

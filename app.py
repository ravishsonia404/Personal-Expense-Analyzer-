from flask import Flask, render_template, request, jsonify
import mysql.connector
import os
import pandas as pd
import numpy as np

# Charts
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# ML
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

# ---------------- GLOBAL DATA ----------------
global_df = None

# ---------------- DATABASE ----------------
import sqlite3

conn = sqlite3.connect("database.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT
)
""")
conn.commit()

# ---------------- FOLDERS ----------------
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("static", exist_ok=True)

# ---------------- PDF ----------------
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
    global global_df

    prediction = None
    total_spent = None
    max_category = None
    pdf_file = None
    message = None

    if request.method == "POST":

        # SAVE USER
        if "name" in request.form:
            name = request.form['name']
            email = request.form['email']

            cursor.execute("INSERT INTO users (name, email) VALUES (%s, %s)", (name, email))
            conn.commit()

            message = "✅ Data Saved!"

        # FILE UPLOAD
        if "file" in request.files:
            file = request.files.get("file")

            if file and file.filename != "":
                filepath = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(filepath)

                data = pd.read_csv(filepath)
                global_df = data  # 👈 STORE DATA FOR CHATBOT

                data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
                data['Amount'] = pd.to_numeric(data['Amount'], errors='coerce')
                data = data.dropna(subset=['Date', 'Amount'])

                total_spent = int(data['Amount'].sum())
                max_category = data.groupby("Category")["Amount"].sum().idxmax()

                # ML
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

                # Charts
                plt.figure()
                monthly.plot(kind='bar')
                plt.savefig("static/chart.png")
                plt.close()

                plt.figure()
                data.groupby("Category")["Amount"].sum().plot(kind='pie', autopct='%1.1f%%')
                plt.savefig("static/pie.png")
                plt.close()

                pdf_file = generate_pdf(prediction, total_spent, max_category)

                message = "✅ File uploaded! You can now chat with your data."

    return render_template("index.html",
                           prediction=prediction,
                           total=total_spent,
                           category=max_category,
                           pdf_file=pdf_file,
                           message=message)

# ---------------- CHATBOT ----------------
@app.route("/chat", methods=["POST"])
def chat():
    global global_df

    data = request.get_json()
    user_msg = data.get("message", "").lower()

    if global_df is None:
        return jsonify({"reply": "Please upload a file first."})

    total = int(global_df['Amount'].sum())
    category = global_df.groupby("Category")["Amount"].sum().idxmax()

    # 🔥 SMART MATCHING
    if any(word in user_msg for word in ["total", "spending", "expense", "overall"]):
        reply = f"Your total spending is ₹{total}"

    elif any(word in user_msg for word in ["category", "most", "highest", "where", "spent"]):
        reply = f"You spent the most in {category}"

    elif any(word in user_msg for word in ["advice", "suggest", "improve"]):
        reply = f"Try reducing expenses in {category} to save more money."

    else:
        reply = "You can ask things like:\n- Where did I spend most?\n- Total expense?\n- Give advice"

    return jsonify({"reply": reply})

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)

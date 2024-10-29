# app.py
from flask import Flask, jsonify, render_template
from dash import Dash
from dashboard.layout import create_layout
import pandas as pd

# Initialize Flask app
server = Flask(__name__)

# Load and preprocess fraud data
fraud_data = pd.read_csv("data/Fraud_Data.csv")

@server.route("/")
def index():
    return render_template("index.html")

@server.route("/api/summary")
def get_summary():
    total_transactions = len(fraud_data)
    total_fraud = fraud_data['is_fraud'].sum()
    fraud_percentage = (total_fraud / total_transactions) * 100
    
    summary = {
        "total_transactions": total_transactions,
        "total_fraud": int(total_fraud),
        "fraud_percentage": fraud_percentage
    }
    return jsonify(summary)

@server.route("/api/fraud_trends")
def get_fraud_trends():
    fraud_trends = fraud_data.groupby("date")["is_fraud"].sum().reset_index()
    fraud_trends["date"] = pd.to_datetime(fraud_trends["date"]).dt.strftime('%Y-%m-%d')
    return fraud_trends.to_json(orient="records")

# Initialize Dash app
app = Dash(__name__, server=server, url_base_pathname='/dashboard/')
app.layout = create_layout(app)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=True)

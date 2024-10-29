# dashboard/callbacks.py
from dash import Input, Output, dcc, html
import dash_bootstrap_components as dbc
import requests
import pandas as pd
import plotly.express as px

def register_callbacks(app):
    @app.callback(
        [Output("total-transactions", "children"),
         Output("total-fraud", "children"),
         Output("fraud-percentage", "children")],
        Input("url", "pathname")
    )
    def update_summary(_):
        response = requests.get("http://localhost:5000/api/summary")
        data = response.json()
        
        return (
            f"Total Transactions: {data['total_transactions']}",
            f"Total Fraud Cases: {data['total_fraud']}",
            f"Fraud Percentage: {data['fraud_percentage']:.2f}%"
        )

    @app.callback(
        Output("fraud-trends-chart", "figure"),
        Input("url", "pathname")
    )
    def update_fraud_trends(_):
        trends_data = pd.read_json(requests.get("http://localhost:5000/api/fraud_trends").text)
        fig = px.line(trends_data, x="date", y="is_fraud", title="Fraud Cases Over Time")
        return fig

    @app.callback(
        Output("device-fraud-chart", "figure"),
        Input("url", "pathname")
    )
    def update_device_fraud_chart(_):
        device_data = fraud_data.groupby("device_type")["is_fraud"].sum().reset_index()
        fig = px.bar(device_data, x="device_type", y="is_fraud", title="Fraud Cases by Device Type")
        return fig

    @app.callback(
        Output("browser-fraud-chart", "figure"),
        Input("url", "pathname")
    )
    def update_browser_fraud_chart(_):
        browser_data = fraud_data.groupby("browser")["is_fraud"].sum().reset_index()
        fig = px.bar(browser_data, x="browser", y="is_fraud", title="Fraud Cases by Browser")
        return fig

    @app.callback(
        Output("fraud-geo-chart", "figure"),
        Input("url", "pathname")
    )
    def update_fraud_geo_chart(_):
        geo_data = fraud_data.groupby("location")["is_fraud"].sum().reset_index()  # Adjust if latitude/longitude available
        fig = px.choropleth(geo_data, locations="location", color="is_fraud",
                            locationmode='country names', title="Fraud Cases by Location")
        return fig

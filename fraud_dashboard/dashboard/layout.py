# dashboard/layout.py
from dash import dcc, html
import dash_bootstrap_components as dbc
from .callbacks import register_callbacks

def create_layout(app):
    layout = dbc.Container([
        dbc.Row([
            dbc.Col(html.H2("Fraud Detection Dashboard"), width=12)
        ]),
        dbc.Row([
            dbc.Col(html.Div(id="total-transactions"), width=4),
            dbc.Col(html.Div(id="total-fraud"), width=4),
            dbc.Col(html.Div(id="fraud-percentage"), width=4),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id="fraud-trends-chart"), width=6),
            dbc.Col(dcc.Graph(id="fraud-geo-chart"), width=6),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id="device-fraud-chart"), width=6),
            dbc.Col(dcc.Graph(id="browser-fraud-chart"), width=6),
        ])
    ])
    
    register_callbacks(app)  # Register the callbacks for interactivity
    return layout

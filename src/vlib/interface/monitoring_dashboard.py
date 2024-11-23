from pathlib import Path

import dash
import pandas as pd
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px

from vlib.config import paths
from vlib.config.const import PREDICTION, MEAN_ABSOLUTE_ERROR, DATETIME

app = dash.Dash(__name__)

def calculate_daily_mae():
    """
    compute mean absolute error per day
    :return: dataframe aggregated per day with MAE
    """
    df = pd.read_csv(Path(paths.PREDICTIONS_CSV))
    df[DATETIME] = pd.to_datetime(df[DATETIME])
    df['date'] = df[DATETIME].dt.date
    df[MEAN_ABSOLUTE_ERROR] = abs(df['count'] - df[PREDICTION])
    daily_mae = df.groupby('date')[MEAN_ABSOLUTE_ERROR].mean().reset_index()
    return daily_mae


app.layout = html.Div([
    html.H1("Daily Mean Absolute Error (MAE)"),
    dcc.Graph(id='mae-graph'),
    dcc.Interval(
        id='interval-component',
        interval=10 * 1000,  # 10 secondes
        n_intervals=0
    )
])

@app.callback(
    Output('mae-graph', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_graph(n):
    daily_mae = calculate_daily_mae()
    fig = px.line(daily_mae, x='date', y='mean_absolute_error', title='Mean Absolute Error (MAE) per day')
    return fig

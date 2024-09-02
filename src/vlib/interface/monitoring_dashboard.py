import dash
import pandas as pd
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px

app = dash.Dash(__name__)

def calculate_daily_mae():
    df = pd.read_csv('data/predictions.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['date'] = df['datetime'].dt.date
    df['mean_absolute_error'] = abs(df['count'] - df['prediction'])
    daily_mae = df.groupby('date')['mean_absolute_error'].mean().reset_index()
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

# Importing necessary libraries
from datetime import datetime as dt

import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px
import yfinance as yf
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

# model
from mod import prediction

def get_stock_price_fig(df):
    fig = px.line(df,
                  x="Date",
                  y=["Close", "Open"],
                  title="Closing and Opening price vs Date")
    
    return fig

def get_more(df):
    df['EWA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    fig = px.scatter(df,
                     x="Date",
                     y="EWA_20",
                     title="Exponential moving Average vs Date")

    fig.update_traces(mode='lines+markers')
    return fig

# Creating a Dash Instance and a server variable
app = dash.Dash(
    __name__, external_stylesheets=["style.css", "https://fonts.googleapis.com/css2?family=Roboto&display=swap"])
server = app.server
# making the web layout using Dash Html components
#  and then storing it in app's layout component
# we mainly need two div for entire layout

#  The first one is for our inputs like stock code, date range selector, number of
# days of forecast and buttons.
# The second division will be for the data plots and company's basic information
# (name, logo, brief intro) only
app.layout = html.Div(
    [

        # item1=
        html.Div(
            [
                html.P("Welcome to Stock Dash APP!", className="start"),
                html.P("here you can see the stock price graph and predict the stock price of next few days ",
                       className="start"),
                html.Div([
                    # stock code input
                    html.P("Input stock code:"),
                    html.Div([
                        dcc.Input(id="dropdown_tickers", type="text"),
                        html.Button("Submit", id='submit'),
                    ],
                        className="form")
                ],
                    className="input-place"),
                html.Div([
                    # Date range picker input
                    dcc.DatePickerRange(id='my-date-picker-range', min_date_allowed=dt(1995, 8, 5),
                                        max_date_allowed=dt.now(),
                                        initial_visible_month=dt.now(),
                                        end_date=dt.now().date()),
                ],
                    className="date"),
                html.Div([
                    # Stock price button
                    html.Button("Stock Price", className="stock-btn", id="stock"),
                    # Indicators button
                    html.Button("Indicators", className="indicators-btn",
                                id="indicators"),
                    # Number of days of forecast input
                    dcc.Input(id="n_days", type="text",
                              placeholder="number of days"),
                    # Forecast button
                    html.Button("Forecast", className="forecast-btn",
                                id="forecast")
                ],
                    className="buttons"),
            ],
            className="nav"),

        # item2,content
        html.Div(
            [
                html.Div(
                    [
                        # Logo
                        html.Img(id="logo"),
                        # company name
                        html.P(id="ticker")
                    ],
                    className="header"),
                # Desription
                html.Div(id="description", className="decription_ticker"),
                # Stock price plot
                html.Div([], id="graphs-content"),
                # indicator plot
                html.Div([], id="main-content"),
                # Forecast plot
                html.Div([], id="forecast-content")
            ],
            className="content"),
    ],
    className="container")

# callback for company info

@app.callback([
    Output("description", "children"),
    Output("logo", "src"),
    Output("ticker", "children"),
    Output("stock", "n_clicks"),
    Output("indicators", "n_clicks"),
    Output("forecast", "n_clicks")
], [Input("submit", "n_clicks")], [State("dropdown_tickers", "value")])
def update_data(n, val):  # Input parameters
    if n == None:
        return "Hey there! Please enter a legitimate stock code to get details.", "https://img.etimg.com/thumb/width-640,height-480,imgsize-189125,resizemode-1,msid-68962645/wealth/stock-market-hitting-new-highs-3-value-stocks-to-invest-in-now/stock-getty.jpg", "Stocks", None, None, None
        # raise PreventUpdate
    else:
        if val == None:
            raise PreventUpdate
        else:
            ticker = yf.Ticker(val)
            inf = ticker.info
            df = pd.DataFrame().from_dict(inf, orient="index").T
            df[['logo_url', 'shortName', 'longBusinessSummary']]
            return df['longBusinessSummary'].values[0], df['logo_url'].values[
                0], df['shortName'].values[0], None, None, None

# callback for stocks graphs
@app.callback([
    Output("graphs-content", "children"),
], [
    Input("stock", "n_clicks"),
    Input('my-date-picker-range', 'start_date'),
    Input('my-date-picker-range', 'end_date')
], [State("dropdown_tickers", "value")])
def stock_price(n, start_date, end_date, val):
    if n == None:
        return [""]
    if val == None:
        raise PreventUpdate
    else:
        if start_date != None:
            df = yf.download(val, str(start_date), str(end_date))
        else:
            df = yf.download(val)
    df.reset_index(inplace=True)
    fig = get_stock_price_fig(df)
    return [dcc.Graph(figure=fig)]

# callback for indicators
@app.callback([
    Output("main-content", "children")],
    [
        Input("indicators", "n_clicks"),
        Input('my-date-picker-range', 'start_date'),
        Input('my-date-picker-range', 'end_date')
    ], [State("dropdown_tickers", "value")]
)
def indicators(n, start_date, end_date, val):
    if n == None:
        return [""]
    if val == None:
        return [""]

    if start_date == None:
        df_more = yf.download(val)
    else:
        df_more = yf.download(val, str(start_date), str(end_date))

    df_more.reset_index(inplace=True)
    fig = get_more(df_more)
    return [dcc.Graph(figure=fig)]

# callback for forecast
@app.callback([Output("forecast-content", "children")],
              [Input("forecast", "n_clicks")],
              [State("n_days", "value"),
               State("dropdown_tickers", "value")])
def forecast(n, n_days, val):
    if n == None:
        return [""]
    if val == None:
        raise PreventUpdate
    fig = prediction(val, int(n_days) + 1)
    return [dcc.Graph(figure=fig)]

if __name__ == '__main__':
    app.run_server(debug=True)

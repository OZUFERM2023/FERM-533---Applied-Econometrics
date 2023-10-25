# Let's start with importing the necessary libraries as we go
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import date
import statsmodels.api as sm
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd

def plot_stock_price(ticker, start_date, end_date):
    # Download stock data from Yahoo Finance
    stock_data = yf.download(ticker, start=start_date, end=end_date)

    # Resample data to monthly frequency
    monthly_data = stock_data['Close'].resample('M').mean()

    # Plot the stock price
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_data, linestyle='-')
    plt.title(f'{ticker} Stock Price (Monthly)')
    plt.xlabel('Date')
    plt.ylabel('Stock Price (USD)')
    plt.grid(True)
    plt.show()
    
def calculate_monthly_returns(ticker, start_date, end_date):
    # Download stock data from Yahoo Finance
    stock_data = yf.download(ticker, start=start_date, end=end_date)

    # Calculate daily returns
    daily_returns = stock_data['Close'].pct_change()

    # Resample daily returns to monthly frequency and drop NaN values
    monthly_returns = daily_returns.resample('M').mean().dropna()

    # Print summary
    print("\nSummary of Monthly Returns:")
    print(monthly_returns.describe())
    
    return monthly_returns

def plot_monthly_returns(ticker, start_date, end_date):
    # Calculate monthly returns
    monthly_returns = calculate_monthly_returns(ticker, start_date, end_date)

    # Plot the monthly returns
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_returns, linestyle='-')
    plt.title(f'{ticker} Monthly Returns')
    plt.xlabel('Date')
    plt.ylabel('Monthly Returns')
    plt.grid(True)
    plt.show()
    

def regress_returns(stock_ticker, benchmark_ticker, start_date, end_date):
    # Download stock and benchmark data from Yahoo Finance
    stock_data = yf.download(stock_ticker, start=start_date, end=end_date)['Close'].pct_change().dropna()
    benchmark_data = yf.download(benchmark_ticker, start=start_date, end=end_date)['Close'].pct_change().dropna()

    # Add a constant term to the independent variable (benchmark)
    X = sm.add_constant(benchmark_data)

    # Fit the regression model, use Ordinary Least Squares
    model = sm.OLS(stock_data, X).fit()

    # Print out the regression results
    print(model.summary())

    # Plot actual vs predicted returns
    plt.figure(figsize=(12, 6))

    # Plot actual returns
    plt.plot(stock_data.index, stock_data, label=f'{stock_ticker} Actual Returns', linestyle='-')

    # Plot predicted returns
    plt.plot(stock_data.index, model.predict(X), label=f'{stock_ticker} Predicted Returns', linestyle='-')

    plt.title(f'{stock_ticker} Returns vs. Benchmark Returns')
    plt.xlabel('Date')
    plt.ylabel('Returns')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def simple_regression(ticker, start_date, end_date):
    # Download stock data from Yahoo Finance
    stock_data = yf.download(ticker, start=start_date, end=end_date)

    # Calculate monthly returns
    stock_data['Returns'] = stock_data['Close'].pct_change()

    # Create a dummy variable for January
    stock_data['Is_January'] = (stock_data.index.month == 1).astype(int)

    # Drop missing values
    stock_data = stock_data.dropna()

    # Run regression
    X = sm.add_constant(stock_data['Is_January'])
    y = stock_data['Returns']
    model = sm.OLS(y, X).fit()

    # Print regression summary
    print(model.summary())

def Seasonality_Dynamic(ticker, start=None, end=None):

    if start == None:
        start = "2020-01-01"
    else:
        start = start
    if end == None:
        end = date.today()
    else:
        end = end
        
    price = yf.download(ticker, start, end)
    df = pd.DataFrame({'return': price['Close'].pct_change().fillna(0)})

    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    df = df[df.index >= df[df.index.month == 1].index[0]]
    df = df[df.index <= df[df.index.month == 12].index[-1]]
    
    # Seasonal data
    seasonal_data = {}
    for year in df.index.year.unique():
        seasonal_data[year] = df[df.index.year == year].reset_index()['return']
    seasonal_data = pd.DataFrame(seasonal_data)
    
   # Monthly Cumulative Returns
    year_long = seasonal_data[-1:].T.dropna().index[0]
    seasonal_data.index = df[df.index.year == year_long].index.strftime('%Y%m')
    seasonal_returns = seasonal_data.dropna(how='all').groupby(seasonal_data.index).cumsum()
    seasonal_returns.reset_index(drop=True, inplace=True)
    seasonal_returns = seasonal_returns.dropna(how='all').mean(axis=1) 
    
    # Monthly Data Summary
    monthly = {}
    for year in df.index.year.unique():
        yeardf = df[df.index.year == year]
        monthly[year] = yeardf.groupby(yeardf.index.month).sum() * 100

    data = pd.concat(monthly, axis=1)
    data.columns = [col[0] for col in data.columns]
    data.index = months

    summary = pd.DataFrame(data.mean(axis=1))
    summary.columns = ['Return %']
    
    # Create a line plot using plotly.graph_objs
    fig = go.Figure()

    # Add a line trace for the summary data
    fig.add_trace(go.Scatter(
        x=summary.index,
        y=summary['Return %'].round(2),
        mode='lines+markers',
        name='Monthly Returns',
        line=dict(color='green'),
        marker=dict(size=8, color='green')
    ))

    # Set plot title and axis labels
    fig.update_layout(
        title=f'Seasonal Chart : {ticker}',
        xaxis=dict(title='Month'),
        yaxis=dict(title='Return %'),
        template='plotly_dark'
    )
    # Add a zero line to the plot
    fig.add_shape(
        type="line",
        x0=summary.index[0],
        y0=0,
        x1=summary.index[-1],
        y1=0,
        line=dict(color="red", dash="dash")
    )

    # Show the plot
    fig.show()
    
    data_df = pd.DataFrame(data.T)
    return data_df

def Seasonality_Static(ticker, start=None, end=None):
    if start is None:
        start = "2020-01-01"
    if end is None:
        end = date.today()

    price = yf.download(ticker, start, end)
    df = pd.DataFrame({'return': price['Close'].pct_change().fillna(0)})

    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    df = df[df.index >= df[df.index.month == 1].index[0]]
    df = df[df.index <= df[df.index.month == 12].index[-1]]

    # Seasonal data
    seasonal_data = {}
    for year in df.index.year.unique():
        seasonal_data[year] = df[df.index.year == year].reset_index()['return']
    seasonal_data = pd.DataFrame(seasonal_data)

    # Monthly Cumulative Returns
    year_long = seasonal_data[-1:].T.dropna().index[0]
    seasonal_data.index = df[df.index.year == year_long].index.strftime('%Y%m')
    seasonal_returns = seasonal_data.dropna(how='all').groupby(seasonal_data.index).cumsum()
    seasonal_returns.reset_index(drop=True, inplace=True)
    seasonal_returns = seasonal_returns.dropna(how='all').mean(axis=1)

    # Monthly Data Summary
    monthly = {}
    for year in df.index.year.unique():
        yeardf = df[df.index.year == year]
        monthly[year] = yeardf.groupby(yeardf.index.month).sum() * 100

    data = pd.concat(monthly, axis=1)
    data.columns = [col[0] for col in data.columns]
    data.index = months

    summary = pd.DataFrame(data.mean(axis=1))
    summary.columns = ['Return %']

    # Create a line plot using matplotlib
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(summary.index, summary['Return %'].round(2), marker='o', color='green', label='Monthly Returns')
    ax.axhline(0, color='red', linestyle='--', label='Zero Return')

    # Set plot title and axis labels
    ax.set(title=f'Seasonal Chart : {ticker}', xlabel='Month', ylabel='Return %')
    ax.legend()
    plt.grid(True)
    plt.show()

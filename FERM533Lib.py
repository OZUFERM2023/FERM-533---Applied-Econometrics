# Let's start with importing the necessary libraries as we go
import yfinance as yf
import matplotlib.pyplot as plt 
import statsmodels.formula.api as smf
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import date

import warnings
warnings.filterwarnings("ignore")

def plot_stock_price(ticker, start_date):
    # Download stock data from Yahoo Finance
    stock_data = yf.download(ticker, start=start_date, interval='1mo')

    # Resample data to monthly frequency
    stock_data['Returns'] = stock_data['Adj Close'].pct_change().dropna()

    # Plot the stock price
    plt.figure(figsize=(12, 6))
    plt.plot(stock_data['Adj Close'], linestyle='-')
    plt.title(f'{ticker} Stock Price (Monthly)')
    plt.xlabel('Date')
    plt.ylabel('Stock Price (USD)')
    plt.grid(True)
    plt.show()
    
def calculate_monthly_returns(ticker, start_date):
    # Download stock data from Yahoo Finance
    stock_data = yf.download(ticker, start=start_date, interval='1mo')

    # Resample data to monthly frequency
    stock_data['Returns'] = stock_data['Adj Close'].pct_change().dropna()

    # Print summary
    print("\nSummary of Monthly Returns:")
    print(stock_data['Returns'].describe())
    
    return stock_data['Returns']

def plot_monthly_returns(ticker, start_date):
    # Calculate monthly returns
    monthly_returns = calculate_monthly_returns(ticker, start_date)

    # Plot the monthly returns
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_returns, linestyle='-')
    plt.title(f'{ticker} Monthly Returns')
    plt.xlabel('Date')
    plt.ylabel('Monthly Returns')
    plt.grid(True)
    plt.show()
    

def regress_returns(stock_ticker, benchmark_ticker, start_date):
    # Download stock and benchmark data from Yahoo Finance
    stock_data = yf.download(stock_ticker, start=start_date, interval='1mo')
    stock_data['Returns'] = stock_data['Adj Close'].pct_change()
    benchmark_data = yf.download(benchmark_ticker, start=start_date, interval='1mo')
    benchmark_data['Returns'] = benchmark_data['Adj Close'].pct_change()
    merged_data = stock_data[['Returns']].join(benchmark_data['Returns'].rename('Benchmark_Returns')).dropna()

    # Fit the regression model, use Ordinary Least Squares
    model = smf.ols("Returns ~ Benchmark_Returns", data=merged_data).fit()

    # Print out the regression results
    print(model.summary())

def conf_int95(stock_ticker, benchmark_ticker, start_date):
    # Download stock and benchmark data from Yahoo Finance
    stock_data = yf.download(stock_ticker, start=start_date, interval='1mo')
    stock_data['Returns'] = stock_data['Adj Close'].pct_change()
    benchmark_data = yf.download(benchmark_ticker, start=start_date, interval='1mo')
    benchmark_data['Returns'] = benchmark_data['Adj Close'].pct_change()
    merged_data = stock_data[['Returns']].join(benchmark_data['Returns'].rename('Benchmark_Returns')).dropna()

    # Fit the regression model, use Ordinary Least Squares
    model = smf.ols("Returns ~ Benchmark_Returns", data=merged_data).fit()
    interval_95 = model.conf_int(alpha=0.05).loc['Benchmark_Returns']
    print(f'95% Confidence Interval for the Coefficient of S&P 500 Returns: \n {interval_95}')
    
def simple_regression(stock_ticker, start_date):
    stock_data = yf.download(stock_ticker, start=start_date, interval='1mo')
    stock_data['Returns'] = stock_data['Adj Close'].pct_change()

    # Create a dummy variable for January
    stock_data['Is_January'] = (stock_data.index.month == 1).astype(int)

    # Run regression
    model = smf.ols("Returns ~ Is_January", data = stock_data.dropna()).fit()

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

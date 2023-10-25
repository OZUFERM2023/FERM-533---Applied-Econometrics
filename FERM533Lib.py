# Let's start with importing the necessary libraries as we go
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import date
import statsmodels.api as sm

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
    

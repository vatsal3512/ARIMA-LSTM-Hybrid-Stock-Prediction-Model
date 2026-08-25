import yfinance as yf
import pandas as pd
import datetime
import matplotlib.pyplot as plt

# Define tickers
company_ticker = 'AAPL'   
index_ticker = '^GSPC'        # S&P 500 Index

#Define date range (last 10 years from today)
end_date = datetime.datetime.today()
start_date = end_date - datetime.timedelta(days=365*10)  # Changed from 30 years to 10 years

#Fetch historical data
company_data = yf.download(company_ticker, start=start_date, end=end_date)
index_data = yf.download(index_ticker, start=start_date, end=end_date)

# Save company data for the hybrid model
company_data.to_csv("AAPL_historical_data.csv")

#Keep only the 'Close' column
company_close = company_data[['Close']].dropna()
index_close = index_data[['Close']].dropna()

#Align both datasets by common dates
data = pd.concat([company_close, index_close], axis=1, join='inner')
data.columns = ['AAPL_Close', 'SP500_Close']

#Remove COVID crash period (2020-02 to 2020-04)
data = data[~((data.index >= '2020-02-15') & (data.index <= '2020-04-30'))] 

#Split data into train and test (80-20)
split_index = int(len(data) * 0.8)
train_data = data[:split_index]
test_data = data[split_index:]

print(f"Total data points: {len(data)}")
print(f"Training data points: {len(train_data)}")
print(f"Testing data points: {len(test_data)}")

#Plot closing prices
plt.figure(figsize=(14,6))
plt.plot(data['AAPL_Close'], label='AAPL')
plt.plot(data['SP500_Close'], label='S&P 500')
plt.axvline(data.index[split_index], color='red', linestyle='--', label='Train-Test Split')
plt.title('Daily Closing Prices (AAPL vs S&P 500)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

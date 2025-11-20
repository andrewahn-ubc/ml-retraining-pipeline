import yfinance as yf

ticker = "SPY"
data = yf.download(ticker, period="1y", progress=False)

print(data)
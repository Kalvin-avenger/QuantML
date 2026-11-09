import yfinance as yf

# df = yf.download(["AAPL", "MSFT"], start="2024-01-01", auto_adjust=True)

# print(df.head())
# print(df.columns)
# print(df.index)
# print(df["Close"].head())


# df = yf.download(["AAPL"], start="2024-01-01", auto_adjust=True)

# print(df.head())
# print(df.columns)
# print(df.index)
# print(df["Close"].head())


df = yf.download(["ZZZZ"], start="2024-01-01", auto_adjust=True)

print(df.head())
print(df.columns)
print(df.index)
print(df["Close"].head())

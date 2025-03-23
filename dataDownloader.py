# coding=utf-8

"""
Financial data downloader from various sources (Alpha Vantage, Yahoo Finance).
"""

import pandas as pd
import requests
import yfinance as yf
import os
from io import StringIO


class AlphaVantage:
    """Downloader for stock market data from the Alpha Vantage API."""

    def __init__(self, apikey='APIKEY'):
        """Initialize with API key and default settings."""
        self.link = 'https://www.alphavantage.co/query'
        self.apikey = apikey
        self.datatype = 'csv'
        self.outputsize = 'full'
        
    def getDailyData(self, marketSymbol, startingDate, endingDate):
        """Download daily stock data from Alpha Vantage."""
        try:
            # Send request to Alpha Vantage API
            payload = {
                'function': 'TIME_SERIES_DAILY_ADJUSTED', 
                'symbol': marketSymbol, 
                'outputsize': self.outputsize, 
                'datatype': self.datatype, 
                'apikey': self.apikey
            }
            response = requests.get(self.link, params=payload)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            # Process CSV response
            csvText = StringIO(response.text)
            data = pd.read_csv(csvText, index_col='timestamp')
            
            # Standardize data format
            data = self._processDataframe(data)
            
            # Filter by date range if provided
            if startingDate != 0 and endingDate != 0:
                data = data.loc[startingDate:endingDate]
                
            return data
            
        except Exception as e:
            print(f"Alpha Vantage download error: {e}")
            return pd.DataFrame()  # Return empty dataframe on error
        
    def getIntradayData(self, marketSymbol, startingDate, endingDate, timePeriod=60):
        """Download intraday data with specified time interval."""
        try:
            # Use valid time periods only
            valid_periods = [1, 5, 15, 30, 60]
            timePeriod = min(valid_periods, key=lambda x:abs(x-timePeriod))
            
            # Send request to Alpha Vantage API
            payload = {
                'function': 'TIME_SERIES_INTRADAY', 
                'symbol': marketSymbol, 
                'outputsize': self.outputsize, 
                'datatype': self.datatype, 
                'apikey': self.apikey, 
                'interval': f"{timePeriod}min"
            }
            response = requests.get(self.link, params=payload)
            response.raise_for_status()
            
            # Process CSV response
            csvText = StringIO(response.text)
            data = pd.read_csv(csvText, index_col='timestamp')
            
            # Standardize data format
            data = self._processDataframe(data)
            
            # Filter by date range if provided
            if startingDate != 0 and endingDate != 0:
                data = data.loc[startingDate:endingDate]
                
            return data
            
        except Exception as e:
            print(f"Alpha Vantage intraday download error: {e}")
            return pd.DataFrame()
    
    def _processDataframe(self, df):
        """Standardize data format from Alpha Vantage."""
        try:
            # Reverse to chronological order
            df = df[::-1]
            
            # Standardize column names
            if 'adjusted_close' in df.columns:
                df['close'] = df['adjusted_close']
                df = df.drop(['adjusted_close', 'dividend_amount', 'split_coefficient'], axis=1, errors='ignore')
            
            # Rename columns and index
            df.index.names = ['Timestamp']
            df = df.rename(columns={
                "open": "Open",
                "high": "High", 
                "low": "Low",
                "close": "Close",
                "volume": "Volume"
            })
            
            # Ensure datetime index
            df.index = pd.to_datetime(df.index)
            return df
            
        except Exception as e:
            print(f"Error processing Alpha Vantage data: {e}")
            return df  # Return original df on error


class YahooFinance:   
    """Downloader for stock market data from Yahoo Finance."""

    def getDailyData(self, marketSymbol, startingDate, endingDate):
        """Download daily stock data from Yahoo Finance."""
        try:
            # Use yfinance to download data
            data = yf.download(marketSymbol, start=startingDate, end=endingDate)
            
            # Standardize format
            if data.empty:
                print(f"No data returned for {marketSymbol}")
                return pd.DataFrame()
                
            # Process data columns
            if 'Adj Close' in data.columns:
                data['Close'] = data['Adj Close']
                data = data.drop('Adj Close', axis=1)
            
            # Ensure proper index name
            data.index.name = 'Timestamp'
            
            # Select only needed columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            data = data[[col for col in required_cols if col in data.columns]]
            
            return data
            
        except Exception as e:
            print(f"Yahoo Finance download error for {marketSymbol}: {e}")
            return pd.DataFrame()


class CSVHandler:
    """Converter between pandas dataframes and CSV files."""
    
    def dataframeToCSV(self, name, dataframe):
        """Save dataframe to CSV file."""
        try:
            path = f"{name}.csv"
            dataframe.to_csv(path)
            print(f"Successfully saved data to {path}")
            return True
        except Exception as e:
            print(f"Error saving CSV {name}: {e}")
            return False


def CSVToDataframe(fileName):
    """Read CSV file and return standardized financial dataframe."""
    try:
        print(f"Reading file: {fileName}")
        
        # Check if file exists
        if not os.path.exists(fileName):
            print(f"File not found: {fileName}")
            return pd.DataFrame()
            
        # Read CSV file
        df = pd.read_csv(fileName)
        
        # Standardize column names
        df.columns = [col.strip() for col in df.columns]
        
        # Map column names to standard form
        column_mappings = {
            'close': 'Close', 
            'adj close': 'Close', 
            'adjusted close': 'Close',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'volume': 'Volume',
            'timestamp': 'Timestamp', 
            'date': 'Timestamp'
        }
        
        # Apply mappings (case-insensitive)
        new_cols = {}
        for col in df.columns:
            if col.lower() in column_mappings:
                new_cols[col] = column_mappings[col.lower()]
                
        if new_cols:
            df = df.rename(columns=new_cols)
                
        # Debug output
        print(f"DEBUG: Headers for {fileName} -> {df.columns.tolist()}")

        # Set datetime index
        if "Timestamp" in df.columns:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"])
            df.set_index("Timestamp", inplace=True)
        else:
            # Try using first column as timestamp
            df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
            df.set_index(df.columns[0], inplace=True)
        
        # Ensure all required columns exist
        required_columns = ['Open', 'High', 'Low', 'Close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"WARNING: Missing columns in {fileName}: {missing_columns}")
            # If Close is missing but we have another price column, use it
            if 'Close' in missing_columns and any(col in df.columns for col in ['Price', 'price']):
                price_col = next(col for col in df.columns if col.lower() == 'price')
                df['Close'] = df[price_col]
                print(f"Using {price_col} as Close price")
            
        # Remove duplicates and resample to daily frequency
        df = df[~df.index.duplicated(keep="first")]
        df = df.resample("D").mean().interpolate(method="linear")
        
        return df
    
    except Exception as e:
        print(f"ERROR reading {fileName}: {str(e)}")
        return pd.DataFrame()
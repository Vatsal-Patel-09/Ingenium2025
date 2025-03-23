import requests
from datetime import datetime, timezone

def to_timestamp(time_input):
    """
    Converts various time inputs to a Unix timestamp (integer seconds since epoch).

    Parameters:
    - time_input: Can be an integer (timestamp), a string ('YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'),
                  or a timezone-aware datetime object (in UTC).

    Returns:
    - int: Unix timestamp in seconds.

    Raises:
    - ValueError: If the input is invalid or cannot be converted.
    """
    if isinstance(time_input, int):
        return time_input
    elif isinstance(time_input, str):
        try:
            # Try parsing as 'YYYY-MM-DD'
            dt = datetime.strptime(time_input, '%Y-%m-%d')
        except ValueError:
            try:
                # Try parsing as 'YYYY-MM-DD HH:MM:SS'
                dt = datetime.strptime(time_input, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                raise ValueError("Invalid date string format. Use 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'.")
        # Assume the date/time is in UTC
        dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    elif isinstance(time_input, datetime):
        if time_input.tzinfo is None:
            raise ValueError("Datetime object must be timezone-aware and set to UTC.")
        return int(time_input.timestamp())
    else:
        raise ValueError("time_input must be an integer, a 'YYYY-MM-DD' string, or a UTC datetime object.")

def get_ohlcv_data(api_key="cvftaa9r01qtu9s6oo3gcvftaa9r01qtu9s6oo40", symbol=None, resolution=None, from_time=None, to_time=None):
    """
    Fetches OHLCV data with timestamps from the Finnhub API with a custom timeline.

    Parameters:
    - api_key (str): Your Finnhub API key.
    - symbol (str): Stock symbol (e.g., 'AAPL' for Apple).
    - resolution (str): Time interval for each data point (e.g., '1' for 1 minute, 'D' for daily).
                        Supported values: 1, 5, 15, 30, 60, D, W, M.
    - from_time: Start time of the timeline. Can be an integer (Unix timestamp), 
                 a string ('YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'), or a timezone-aware 
                 datetime object (in UTC).
    - to_time: End time of the timeline. Same types as from_time.

    Returns:
    - list of dict: Each dictionary contains 'timestamp', 'open', 'high', 'low', 'close', 'volume'.
                    Returns an empty list if no data is available or if the request fails.
    """
    # Convert from_time and to_time to timestamps
    try:
        from_timestamp = to_timestamp(from_time)
        to_timestamp = to_timestamp(to_time)
    except ValueError as e:
        raise ValueError(f"Invalid time input: {e}")

    # Construct the API URL with the provided parameters
    url = f"https://finnhub.io/api/v1/stock/candle?symbol={symbol}&resolution={resolution}&from={from_timestamp}&to={to_timestamp}&token={api_key}"
    
    # Send the GET request to the Finnhub API
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises an exception for HTTP errors
    except requests.exceptions.RequestException as e:
        raise Exception(f"API request failed: {e}")
    
    # Parse the JSON response
    data = response.json()
    
    # Check if the response is successful and contains required data
    if data.get('s') != 'ok':
        return []  # Return empty list if no data or an error occurs
    if not all(key in data for key in ['t', 'o', 'h', 'l', 'c', 'v']):
        return []  # Return empty list if any OHLCV or timestamp data is missing
    
    # Combine the OHLCV data and timestamps into a list of dictionaries
    ohlcv_data = [
        {
            'timestamp': t,  # Unix timestamp for the start of the interval
            'open': o,       # Opening price
            'high': h,       # Highest price
            'low': l,        # Lowest price
            'close': c,      # Closing price
            'volume': v      # Trading volume
        }
        for t, o, h, l, c, v in zip(data['t'], data['o'], data['h'], data['l'], data['c'], data['v'])
    ]
    
    return ohlcv_data

# Example usage:
# Replace 'your_api_key_here' with your actual Finnhub API key
# api_key = 'your_api_key_here'

# Using date strings
# data = get_ohlcv_data(api_key, 'AAPL', 'D', '2021-01-01', '2021-02-01')

# Using timestamps
# data = get_ohlcv_data(api_key, 'AAPL', 'D', 1609459200, 1612137600)

# Using datetime objects
# from datetime import datetime, timezone
# start = datetime(2021, 1, 1, tzinfo=timezone.utc)
# end = datetime(2021, 2, 1, tzinfo=timezone.utc)
# data = get_ohlcv_data(api_key, 'AAPL', 'D', start, end)

# print(data)
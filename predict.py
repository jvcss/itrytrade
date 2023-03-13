import datetime

import enum

import pytz
import requests
import pandas as pd

from sklearn.preprocessing import StandardScaler

import numpy as np

import plotly.graph_objs as go
from plotly.subplots import make_subplots

import finplot as fplt

INTERVALS = {
    '1m': {'freq': '1min', 'periods': [5]},
    '5m': {'freq': '5min', 'periods': [5]},
    '15m': {'freq': '15min', 'periods': [4, 8]},
    '30m': {'freq': '30min', 'periods': [8]},
    '1h': {'freq': 'H', 'periods': [1, 6, 24]},
    '2h': {'freq': '2H', 'periods': [12]},
    '4h': {'freq': '4H', 'periods': [6]},
    '6h': {'freq': '6H', 'periods': [4]},
    '12h': {'freq': '12H', 'periods': [2]},
    '1d': {'freq': 'D', 'periods': [1, 7]},
}

class ProcessType(enum.Enum):
    STANDARD = 1
    SCALER = 2

class Algorithm(enum.Enum):
    DECISION_TREE = 1
    LINEAR_REGRESSION = 2
    RANDOM_FOREST = 3
    SUPPORT_VECTOR_MACHINE = 4

class DataCollector:
    def __init__(self, base_coin, quote_coin, interval='1d'):
        self.base_url = 'https://api.binance.com/api/v3/klines'
        self.base_coin = base_coin
        self.quote_coin = quote_coin
        self.interval = interval

    def fetch_data(self):
        """Fetch historical data for the specified time range"""
        # Calculate start and end times

        endpoint = f'?symbol={self.base_coin}{self.quote_coin}&interval={self.interval}'
        # code to retrieve data from Binance API using the base_coin, quote_coin and interval arguments
        url = self.base_url + endpoint
        response = requests.get(url)
        data = response.json()
        return data
    
    def specific_fetch_data(self, time_range):
        """Fetch historical data for the specified time range"""
        # Calculate start and end times
        if len(time_range) == 1:
            start_time = int((time_range[0] - datetime.datetime(1970, 1, 1, tzinfo=pytz.UTC)).total_seconds() * 1000)
            end_time = int(datetime.datetime.now().timestamp() * 1000)
        else:
            start_time = int((time_range[0] - datetime.datetime(1970,1,1, tzinfo=pytz.UTC)).total_seconds() * 1000)
            end_time = int((time_range[1] - datetime.datetime(1970,1,1, tzinfo=pytz.UTC)).total_seconds() * 1000)

        #end_time = int((time_range[1] - datetime.datetime(1970,1,1)).total_seconds() * 1000)
        
        # Fetch data
        endpoint = f'?symbol={self.base_coin}{self.quote_coin}&interval={self.interval}&startTime={start_time}&endTime={end_time}'
        url = self.base_url + endpoint
        response = requests.get(url)
        data = response.json()
        return data

class DataPreprocessor:
    def __init__(self, data=None):
        self.data = data

    def preprocess_data_scaler(self, data=None):
        if self.data is None:
            raise ValueError("No data provided to preprocess.")
        if data is None:
            data = self.data
            data = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 
            'volume','close_time','quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
            'taker_buy_quote_asset_volume', 'ignore'])
        else:
            data = np.vstack([self.data, data])
            data = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 
            'volume','close_time','quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
            'taker_buy_quote_asset_volume', 'ignore'])
        data.set_index('timestamp', inplace=True)

        # calculate simple moving averages
        short_sma = data['close'].rolling(window=20).mean()
        long_sma = data['close'].rolling(window=50).mean()

        # calculate exponential moving averages
        short_ema = data['close'].ewm(span=20, adjust=False).mean()
        long_ema = data['close'].ewm(span=50, adjust=False).mean()

        # calculate standard deviation and bollinger bands
        std = data['close'].rolling(window=20).std()
        upper_band = short_sma + 2 * std
        lower_band = short_sma - 2 * std

        # create new data frame with technical indicators and target variable
        processed_data = pd.DataFrame({
            'open': data['open'],
            'high': data['high'],
            'low': data['low'],
            'close': data['close'],
            'short_sma': short_sma,
            'long_sma': long_sma,
            'short_ema': short_ema,
            'long_ema': long_ema,
            'upper_band': upper_band,
            'lower_band': lower_band
        }, index=data.index)

        # remove rows with missing values
        processed_data.dropna(inplace=True)

        X = processed_data.drop('close', axis=1)
        y = processed_data['close']
        
        # normalize X using StandardScaler
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

        return X, y, processed_data['close']

    def preprocess_data(self, data=None):
        if self.data is None:
            raise ValueError("No data provided to preprocess.")
        if data is None:
            data = self.data
        else:
            data = np.vstack([self.data, data])

        # Convert the data to a pandas dataframe
        df = pd.DataFrame(self.data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
        df = df.drop(['close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'], axis=1)


        # Convert the timestamp to a datetime object and set it as the index
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')
        df = df.astype(float)
        df = df.resample('1h').mean().dropna()
        
        X = df.drop('close', axis=1)
        y = df['close']
        return X, y
    
    def generate_prediction_data(self, last_hour,process_type=ProcessType.STANDARD):
        if self.data is None:
            raise ValueError("No data provided to preprocess.")
        next_hour = last_hour + pd.Timedelta(hours=1)
        last_row = self.data[-1]
        next_hour_data = [
            [next_hour.timestamp() * 1000, last_row[1], last_row[2], last_row[3], last_row[4], 0, 0, 0, 0, 0, 0,0]
        ]
        if process_type == ProcessType.SCALER:
            X, _, _ = self.preprocess_data_scaler(next_hour_data)
        elif process_type == ProcessType.STANDARD:
            X, _, = self.preprocess_data(next_hour_data)
        return X

class StockPredictor:
    def __init__(self, algorithm=Algorithm.DECISION_TREE):
        self.algorithm = algorithm

    def select_model(self, X_train, y_train):
        if self.algorithm == Algorithm.DECISION_TREE:
            from sklearn.tree import DecisionTreeRegressor
            model = DecisionTreeRegressor()
        elif self.algorithm == Algorithm.LINEAR_REGRESSION:
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
        elif self.algorithm == Algorithm.RANDOM_FOREST:
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor()
        elif self.algorithm == Algorithm.SUPPORT_VECTOR_MACHINE:
            from sklearn.svm import SVR
            model = SVR()

        model.fit(X_train, y_train)

        return model
    
    def train_model(self, X_train, y_train):
        model = self.select_model(X_train, y_train)
        model.fit(X_train, y_train)
        self.model = model
    
    def predict(self, X_test):
        return self.model.predict(X_test)

class DataVisualizer:
    def __init__(self, data=None):
        self.data = data
    def convert_numeric(self, val):
        try:
            return float(val)
        except ValueError:
            return None
    

    def view_data(self, predictions):
        # Prepare data
        df = pd.DataFrame(self.data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'num_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')
        df = df.applymap(self.convert_numeric)  # convert non-numeric values
        
        # Remove rows with missing values
        df = df.dropna()

        # Prepare predictions
        pred_df = pd.DataFrame({'timestamp': df.index, 'close': predictions})
        pred_df = pred_df.set_index('timestamp')
        
        # Create finplot window
        ax, ax2, ax3 = fplt.create_plot('Historical Data and Predictions', rows=3, maximize=False)

        # Remove y-axis values from second and third rows
        ax2.getAxis('right').style['showValues'] = False
        ax3.getAxis('right').style['showValues'] = False

        # Set labels for y-axes
        ax.setLabel('right', 'Price (USD)')
        ax2.setLabel('right', 'Correlation')
        ax3.setLabel('right', 'Volume')

        # Plot candlesticks
        candles = df[['open', 'close', 'high', 'low']]
        fplt.candlestick_ochl(candles, ax=ax)

        # Plot predicted prices
        fplt.plot(pred_df['close'], ax=ax, color='green', width=1)

        # Plot volume
        fplt.volume_ocv(df[['open', 'close', 'volume']], ax=ax3)

        fplt.show()
    
    def visualize_data(self, predictions):
        trace = go.Candlestick(x=[pd.to_datetime(data[0], unit='ms').strftime('%Y-%m-%d %H:%M:%S') for data in self.data],
                               open=[float(data[1]) for data in self.data],
                               high=[float(data[2]) for data in self.data],
                               low=[float(data[3]) for data in self.data],
                               close=[float(data[4]) for data in self.data],
                               name='Actual')
        trace2 = go.Candlestick(x=[pd.to_datetime(data[0], unit='ms').strftime('%Y-%m-%d %H:%M:%S') for data in self.data],
                                open=[float(data[1]) for data in self.data],
                                high=[float(data[2]) for data in self.data],
                                low=[float(data[3]) for data in self.data],
                                close=predictions,
                                increasing=dict(line=dict(color='#00FF00')),
                                decreasing=dict(line=dict(color='#FF0000')),
                                name='Predicted')
        fig = make_subplots(rows=1, cols=1)
        fig.append_trace(trace, 1, 1)
        fig.append_trace(trace2, 1, 1)
        fig.update_layout(title='BTC/USDT Price Prediction',
                          yaxis_title='Price (USDT)',
                          xaxis_title='Timestamp',
                          xaxis_rangeslider_visible=False)
        fig.show()

def predict_next_closing_price(data_collector, preprocessor, predictor, visualizer, time_range):
    # Fetch data
    #data = data_collector.specific_fetch_data(time_range)
    data = data_collector.fetch_data()

    # Preprocess data
    X, y = preprocessor.preprocess_data(data)

    # Train model
    predictor.train_model(X, y)

    # Generate prediction data
    start_date = time_range[-1]-time_range.freq
    
    next_data = preprocessor.generate_prediction_data_ranged(start_date,time_range[-1], interval=data_collector.interval)
    next_X, _ = preprocessor.preprocess_data(next_data)

    # Predict next closing price
    predicted_closing_price = predictor.predict(next_X)
    print(f"Predicted closing price for {time_range[-1]}: {predicted_closing_price[0]}")

    # Visualize data
    visualizer.visualize_data(data, predicted_closing_price)

#if __name__ == '__main__':
#    pro()#fineeralized version of functions of finance objects


import predict as PRE
import pandas as pd

from sklearn import metrics #example

def example():
    #Collects
    collector = PRE.DataCollector(base_coin='BTC', quote_coin='USDT', interval='1h')
    data = collector.fetch_data()

    #Preprocesses
    preprocessor = PRE.DataPreprocessor(data)
    X, y = preprocessor.preprocess_data()
    
    #Trains
    predictor = PRE.StockPredictor(algorithm=PRE.Algorithm.DECISION_TREE)
    predictor.train_model(X_train=X, y_train=y)

    #Generates prediction
    predictions = predictor.predict(X_test=X)

    # calculate Mean squared error regression loss
    mse = metrics.mean_squared_error(y_true=y, y_pred=predictions)
    print("Mean Squared Error (MSE): ", mse)
    
    # visualize the data
    visualizer = PRE.DataVisualizer(data)
    visualizer.view_data(predictions)

def pro():
    # Collect data
    data_collector = PRE.DataCollector(base_coin='BTC', quote_coin='USDT', interval='1h')
    data = data_collector.fetch_data()

    # Preprocess and scale the data
    preprocessor = PRE.DataPreprocessor(data)
    X_standard, y_standard = preprocessor.preprocess_data()
    X_scaled, y_scaled, _ = preprocessor.preprocess_data_scaler()

    # Train a decision tree model on the standardized data
    predictor_standard = PRE.StockPredictor(algorithm=PRE.Algorithm.DECISION_TREE)
    predictor_standard.train_model(X_standard, y_standard)

    # Train a decision tree model on the scaled data
    predictor_scaled = PRE.StockPredictor(algorithm=PRE.Algorithm.DECISION_TREE)
    predictor_scaled.train_model(X_scaled, y_scaled)

    # Generate predictions for the next hour
    last_hour = X_standard.iloc[-1].name
    next_hour_X_standard = preprocessor.generate_prediction_data(last_hour,process_type=PRE.ProcessType.STANDARD)
    next_hour_X_scaled = preprocessor.generate_prediction_data(last_hour,process_type=PRE.ProcessType.SCALER)

    # Predict the closing price using the models trained on the standardized and scaled data
    predicted_closing_price_standard = predictor_standard.predict(next_hour_X_standard)
    predicted_closing_price_scaled = predictor_scaled.predict(next_hour_X_scaled)

    next_hour = last_hour + pd.Timedelta(hours=1)
    print(f"Predicted closing price for {next_hour} (standardized data): {predicted_closing_price_standard[0]}")
    print(f"Predicted closing price for {next_hour} (scaled data): {predicted_closing_price_scaled[0]}")

    # Visualize the data
    visualizer = PRE.DataVisualizer(data)
    visualizer.visualize_data(predicted_closing_price_standard)



if __name__ == '__main__':
    #PRE.example()
    pro()
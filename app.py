pip install flask yfinance pandas numpy scikit-learn keras
from flask import Flask, render_2021news, request
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import yfinance as yf

app = Flask(__name__)

# 获取股票数据
def get_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# 数据预处理
def preprocess_data(data, time_step=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i-time_step:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler

# 构建LSTM模型
def build_lstm_model(X):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# 训练模型
def train_model(model, X, y):
    model.fit(X, y, batch_size=1, epochs=1)
    return model

# 预测
def predict(model, X, scaler):
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)
    return predictions

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ticker = request.form['ticker']
        start_date = request.form['start_date']
        end_date = request.form['end_date']
        
        data = get_stock_data(ticker, start_date, end_date)
        X, y, scaler = preprocess_data(data)
        model = build_lstm_model(X)
        model = train_model(model, X, y)
        
        predictions = predict(model, X, scaler)
        
        # 将预测结果转换为DataFrame
        dates = data.index[60:].tolist()
        actual_prices = data['Close'].values[60:].tolist()
        predicted_prices = predictions.flatten().tolist()
        
        result = pd.DataFrame({
            'Date': dates,
            'Actual Price': actual_prices,
            'Predicted Price': predicted_prices
        })
        
        return render_template('result.html', table=result.to_html(index=False))
    
    return render_template('kkk.html')

if __name__ == '__main__':
    app.run(debug=True)
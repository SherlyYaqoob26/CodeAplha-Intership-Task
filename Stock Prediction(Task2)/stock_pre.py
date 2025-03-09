# Install required packages
# Run the following commands in your terminal:
# pip install protobuf==3.20.*
# pip install streamlit

# Importing libraries
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import numpy as np
from sklearn.preprocessing import MinMaxScaler
# pip install tensorflow
# pip install tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


#streamlit app
st.title('Stock Price Prediction')
@st.cache_data
def load_data(file_path="TATAMOTORS.NS.csv"):
    df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
    
    # Calculate 50-day Simple Moving Average (SMA)
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # Calculate Daily Percentage Changes
    df['Daily Changes'] = df['Close'].pct_change()
    
    # Drop missing values (from rolling mean and pct_change)
    df.dropna(inplace=True)
    
    return df

#loading & reading data
df=pd.read_csv('TATAMOTORS.NS.csv')
print(df.head(n=10))

# Feature Engineering 
# adding two new features to the data frame to smooth the data and to 
# calculate the daily percentage change in stock prices.
# calaulating moving average for 50 days 
df['SMA_50'] = df['Close'].rolling(window=50).mean()
# calculating the daily percentage change in stock prices.
df['Daily_Change'] = df['Close'].pct_change()
# df['RSI'] = compute_rsi(df['Close'])
df.dropna(inplace=True)

# RSI Calculation
def compute_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df['RSI'] = compute_rsi(df['Close'])

# Streamlit App Layout
st.title("📈 Stock Price Analysis with Streamlit")

# splitting the data into training and testing data
# Split data (70% train, 30% test)
train_size = int(len(df) * 0.7)
train_data = df.iloc[:train_size]
test_data = df.iloc[train_size:]

# scaling the data
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.impute import SimpleImputer

# Replace infinite values with NaN
train_data.replace([float('inf'), -float('inf')], np.nan, inplace=True)
test_data.replace([float('inf'), -float('inf')], np.nan, inplace=True)

# Impute NaN values with the mean
imputer = SimpleImputer(strategy='mean')
train_data[['Daily_Change', 'RSI']] = imputer.fit_transform(train_data[['Daily_Change', 'RSI']])
test_data[['Daily_Change', 'RSI']] = imputer.transform(test_data[['Daily_Change', 'RSI']])

# Normalizing the data
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data[['Close', 'SMA_50', 'Daily_Change', 'RSI']])
test_scaled = scaler.transform(test_data[['Close', 'SMA_50', 'Daily_Change', 'RSI']])

# Building the LSTM model
def create_sequences(data, time_steps=60):
    X_seq, y_seq = [], []
    for i in range(len(data) - time_steps):
        X_seq.append(data[i:i+time_steps])
        y_seq.append(data[i+time_steps, 0])  # Predict 'Close' price
    return np.array(X_seq), np.array(y_seq)

# Prepare sequences
time_steps = 60
X_train_seq, y_train_seq = create_sequences(train_scaled)
X_test_seq, y_test_seq = create_sequences(test_scaled)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Define the LSTM model

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])))
model.add(Dropout(0.3))  # Increased from 0.2 to 0.3
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.3))  # Increased dropout
model.add(Dense(25))
model.add(Dense(1))

# compiling the model
model.compile(optimizer='adam', loss='mse')

history = model.fit(X_train_seq, y_train_seq, epochs=30, batch_size=32, validation_data=(X_test_seq, y_test_seq), verbose=1)
y_pred = model.predict(X_test_seq)


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(y_test_seq, label="Actual Prices", color='blue')
plt.plot(y_pred, label="Predicted Prices", color='red')
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.legend()
plt.title("LSTM Predictions vs Actual Stock Prices")
plt.show()

# Plotting the training history
st.subheader('Training History')

fig_loss, ax_loss = plt.subplots()
ax_loss.plot(history.history['loss'], label='Train Loss', color='blue')
ax_loss.plot(history.history['val_loss'], label='Validation Loss', color='orange')
ax_loss.set_title('Training Loss vs Validation Loss')
ax_loss.set_ylabel('Loss')
ax_loss.set_xlabel('Epochs')
ax_loss.legend()
st.pyplot(fig_loss)

# Plotting Results 
st.subheader("Stock Price Prediction")
fig, ax = plt.subplots(figsize=(10, 6))  
ax.plot(y_test_seq,label='Actual Price', color='blue')
ax.plot(y_pred, label='Predicted Price', color='red')
ax.set_xlabel('Days')
ax.set_ylabel('Stock Price')
ax.set_title('Stock Price Prediction')
ax.legend()
st.pyplot(fig)
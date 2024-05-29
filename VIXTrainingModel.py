import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

class VIXTrainingModel:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.model = None

    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        self.data['DATE'] = pd.to_datetime(self.data['DATE'])
        self.data.set_index('DATE', inplace=True)

    def preprocess_data(self):
        self.data['SMA_10'] = self.data['CLOSE'].rolling(window=10).mean()

    def engineer_features(self):
        poly = PolynomialFeatures(degree=2)
        critical_points = self.data['CLOSE'].values.reshape(-1, 1)
        poly_features = poly.fit_transform(critical_points)
        self.data = pd.concat([self.data, pd.DataFrame(poly_features)], axis=1)

        scaler = StandardScaler()
        self.data = pd.DataFrame(scaler.fit_transform(self.data), columns=self.data.columns)

    def split_data(self):
        X = self.data.drop('CLOSE', axis=1).values
        y = self.data['CLOSE'].values
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def build_model(self, input_shape):
        self.model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            LSTM(50),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse')

    def train_model(self, X_train, y_train, X_test, y_test):
        self.model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), verbose=1)

    def evaluate_model(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        return predictions

# Usage
file_path = 'VIX_History.csv'
vix_model = VIXTrainingModel(file_path)
vix_model.load_data()
vix_model.preprocess_data()
vix_model.engineer_features()
X_train, X_test, y_train, y_test = vix_model.split_data()
input_shape = (X_train.shape[1], 1)
vix_model.build_model(input_shape)
vix_model.train_model(X_train, y_train, X_test, y_test)
predictions = vix_model.evaluate_model(X_test, y_test)
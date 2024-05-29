import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from keras.utils import plot_model

class VIXTrainingModel:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        self.model = None

    def preprocess_data(self):
        self.data['DATE'] = pd.to_datetime(self.data['DATE'])
        self.data.set_index('DATE', inplace=True)

        # Feature Engineering
        self.data['SMA_10'] = self.data['CLOSE'].rolling(window=10).mean()

        # Polynomial Features for critical points
        poly = PolynomialFeatures(degree=2)
        critical_points = self.data['CLOSE'].values.reshape(-1, 1)
        poly_features = poly.fit_transform(critical_points)

        # Combine polynomial features with original data
        self.data = np.hstack((self.data.values, poly_features))

        # Scaling the data
        scaler = StandardScaler()
        self.data = scaler.fit_transform(self.data)

        # Prepare data for LSTM
        X = self.data[:, :-1]  # all columns except the last
        y = self.data[:, 0]  # target variable (e.g., 'CLOSE')

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

    def plot_model_architecture(self, file_path):
        plot_model(self.model, to_file=file_path, show_shapes=True, show_layer_names=True)

# Usage
file_path = 'VIX_History.csv'
vix_model = VIXTrainingModel(file_path)
X_train, X_test, y_train, y_test = vix_model.preprocess_data()
vix_model.build_model(input_shape=(X_train.shape[1], 1))  # Update input_shape based on actual data
vix_model.train_model(X_train, y_train, X_test, y_test)
predictions = vix_model.evaluate_model(X_test, y_test)

# Plot the model architecture
vix_model.plot_model_architecture('model_architecture.png')
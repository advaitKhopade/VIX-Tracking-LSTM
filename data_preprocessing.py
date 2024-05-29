import pandas as pd

class DataPreprocessor:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        return self.data

    def handle_missing_values(self):
        print(f"Missing values in the data: {self.data.isnull().sum().sum()}")
        # Handle missing values if any (e.g., drop rows, interpolate, etc.)
        # self.data = self.data.dropna()

    def preprocess_data(self):
        self.data['DATE'] = pd.to_datetime(self.data['DATE'])
        self.data = self.data.set_index('DATE')
        self.data['SMA_10'] = self.data['CLOSE'].rolling(window=10).mean()
        return self.data
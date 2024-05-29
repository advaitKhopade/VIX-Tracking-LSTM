import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class DataExplorer:
    def __init__(self, data):
        self.data = data

    def describe_data(self):
        print(self.data.describe())

    def plot_data(self):
        plt.figure(figsize=(16, 6))
        plt.plot(self.data.index, self.data['CLOSE'])
        plt.xlabel('Date')
        plt.ylabel('VIX Close')
        plt.title('VIX Closing Prices Over Time')
        plt.show()

    def plot_distribution(self):
        plt.figure(figsize=(8, 6))
        sns.histplot(self.data['CLOSE'], kde=True)
        plt.xlabel('VIX Close')
        plt.ylabel('Count')
        plt.title('Distribution of VIX Closing Prices')
        plt.show()

def main():
    # Load the data
    data = pd.read_csv('VIX_History.csv')
    data['DATE'] = pd.to_datetime(data['DATE'])
    data.set_index('DATE', inplace=True)

    # Create an instance of the DataExplorer class and pass the data
    explorer = DataExplorer(data)

    # Use the methods in the DataExplorer class
    explorer.describe_data()
    explorer.plot_data()
    explorer.plot_distribution()

if __name__ == "__main__":
    main()
import matplotlib.pyplot as plt
import seaborn as sns


class VisualizeData:
    def __init__(self, data):
        self.data = data
        self.train_data = None

    def load_data(self):
        # load training dataset and select only numeric columns
        self.train_data = self.data.select_dtypes(include=['int64', 'float64'])

    def plot_distribution(self):
        # plot distribution of Sale Price
        plt.figure(figsize=(10, 6))
        sns.histplot(self.train_data['SalePrice'])
        plt.title('Distribution of Sale Price')
        plt.xlabel('Sale Price in USD$')
        plt.ylabel('Frequency')
        plt.show()

    def plot_predictions_without_actuals(self, predictions, title):
        # Plotting a histogram of predicted sale prices
        plt.figure(figsize=(10, 6))
        sns.histplot(predictions)
        plt.title(title)
        plt.xlabel('Sale Price Predicted in USD$')
        plt.ylabel('Frequency')
        plt.show()

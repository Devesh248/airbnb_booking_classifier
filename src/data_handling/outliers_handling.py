import pandas as pd
import numpy as np


class OutlierHandling:
    """
    Class to handle outliers detection and handling
    """
    def __init__(self, data, numerical_columns):
        """
        Args:
            data (pd.DataFrame): Input datagram on which outlier detection will applied.
            numerical_columns (list): List of numerical columns in dataframe.
        """
        self.data = data
        self.numerical_columns = numerical_columns

    def handle_outliers(self):
        """
        It will detect the outlier using IQR and handle them by using clipping method
        """

        for col in self.numerical_columns:
            q1 = np.percentile(self.data[col].dropna(), 25, method='midpoint')
            q3 = np.percentile(self.data[col].dropna(), 75, method='midpoint')
            iqr = q3 - q1

            upper = q3 + 1.5 * iqr
            lower = q1 - 1.5 * iqr

            # Clipping the outliers
            self.data[col] = np.where(self.data[col] < lower, lower, self.data[col])
            self.data[col] = np.where(self.data[col] > upper, upper, self.data[col])

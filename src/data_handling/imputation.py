import joblib
import os
from sklearn.impute import SimpleImputer
from src.common.constants import *

class Imputation:
    """
    Class to handle missing values for numerical and categorical columns
    """
    def __init__(self, data, numerical_columns, categorical_columns):
        """
        Args:
            data (pd.DataFrame): Input datagram on which outlier detection will applied.
            numerical_columns (list): List of numerical columns in dataframe.
            categorical_columns (list): List of categorical columns in dataframe.
        """
        self.data = data
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns

    def handle_imputation(self):
        """
        Handle imputation for numerical and categorical columns
        """

        # Handling numerical column missing values
        numerical_imp = SimpleImputer(strategy='mean')
        numerical_imp_df = numerical_imp.fit_transform(self.data[self.numerical_columns])
        for index, col in enumerate(self.numerical_columns):
            self.data[col] = numerical_imp_df[:, index]
        joblib.dump(numerical_imp, os.path.join(model_dir, "numerical_imp.joblib"))

        # Handling categorical column missing values
        categorical_imp = SimpleImputer(strategy='most_frequent')

        categorical_imp_df = categorical_imp.fit_transform(self.data[self.categorical_columns])
        for index, col in enumerate(self.categorical_columns):
            self.data[col] = categorical_imp_df[:, index]
        joblib.dump(categorical_imp, os.path.join(model_dir, "categorical_imp.joblib"))

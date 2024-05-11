import os
import joblib
import pandas as pd
from src.common.constants import *


class Prediction:
    """
    Class to perform single and batch predictions
    """

    def __init__(self):
        """
        Load the latest transformer and model
        """

        self.class2id = joblib.load(os.path.join(model_dir, 'class2id.joblib'))
        self.id2class = joblib.load(os.path.join(model_dir, 'id2class.joblib'))

        self.numerical_columns = joblib.load(os.path.join(model_dir, 'numerical_cols.joblib'))
        self.categorical_columns = joblib.load(os.path.join(model_dir, 'categorical_cols.joblib'))

        self.all_columns = self.numerical_columns + self.categorical_columns

        # Load the latest Imputer
        numerical_imp_list = [t for t in os.listdir(model_dir) if "numerical_imp" in t]
        numerical_imp_name = sorted(numerical_imp_list)[-1]
        self.numerical_imp = joblib.load(os.path.join(model_dir, numerical_imp_name))

        categorical_imp_list = [t for t in os.listdir(model_dir) if "categorical_imp" in t]
        categorical_imp_name = sorted(categorical_imp_list)[-1]
        self.categorical_imp = joblib.load(os.path.join(model_dir, categorical_imp_name))

        # Load the latest transformer
        transformer_list = [t for t in os.listdir(model_dir) if "transformer" in t]
        transformer_name = sorted(transformer_list)[-1]
        self.transformer = joblib.load(os.path.join(model_dir, transformer_name))

        # Load the latest model
        model_list = [t for t in os.listdir(model_dir) if "svc_model" in t]
        model_name = sorted(model_list)[-1]
        self.model = joblib.load(os.path.join(model_dir, model_name))

    def predict(self, df):
        numerical_imp_df = self.numerical_imp.transform(df[self.numerical_columns])
        for index, col in enumerate(self.numerical_columns):
            df[col] = numerical_imp_df[:, index]

        categorical_imp_df = self.categorical_imp.transform(df[self.categorical_columns])
        for index, col in enumerate(self.categorical_columns):
            df[col] = categorical_imp_df[:, index]

        x = self.transformer.transform(df)

        predictions = self.model.predict(x)
        results = []

        for i in range(len(predictions)):
            pred_label = self.id2class[predictions[i]]
            results.append(pred_label)

        return results

    def single_prediction_handler(self, data, n_data):
        """
        Args:
              data (pd.DataFrame): Data in dictionary form
              n_data (int): Number of data point
        """
        if n_data == 1:
            df = pd.DataFrame(data, index=[0])
        else:
            df = pd.DataFrame(data)
        df = df[self.all_columns]
        return self.predict(df)

    def batch_prediction_handler(self, batch_file_path):
        df = pd.read_csv(batch_file_path, encoding='latin-1')
        df = df[self.all_columns]
        return self.predict(df)








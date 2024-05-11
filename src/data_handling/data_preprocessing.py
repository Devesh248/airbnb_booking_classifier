import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from src.common.constants import *
from src.data_handling.outliers_handling import OutlierHandling
from src.data_handling.imputation import Imputation
from src.feature_engineering.feature_engineering import *


class DataPreprocessing:
    """
    Class to handle all the tasks related to data pre-processing
    """
    def __init__(self):
        dataset = pd.read_csv(os.path.join(raw_data_dir, data_file_name), encoding="latin-1")
        self.X = dataset.drop(target_column_name, axis=1)
        self.y = dataset[target_column_name]
        numerical_cols = list(self.X.select_dtypes(include=['number']))
        categorical_cols = list(self.X.select_dtypes(include=['object', 'category']))

        self.class2id = dict()
        self.id2class = dict()

        self.class_weights_dict = None

        for index, country in enumerate(self.y.unique()):
            self.class2id[country] = index
            self.id2class[index] = country

        self.outlier_handler = OutlierHandling(self.X, numerical_cols)
        self.impute_handler = Imputation(self.X, numerical_cols, categorical_cols)
        self.feature_handler = FeatureEngineering(self.X, self.y, numerical_cols, categorical_cols, self.class2id)

    def preprocess(self):
        """
        Perform data preprocessing
        Returns:
            tuple: Tuple returning the X and y
        """
        class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(self.y), y=self.y)

        for cls, index in self.class2id.items():
            self.class_weights_dict[index] = class_weights[index]

        self.outlier_handler.handle_outliers()
        self.impute_handler.handle_imputation()
        self.feature_handler.prepare_features()

        return self.X, self.y, self.class_weights_dict, self.class2id, self.id2class



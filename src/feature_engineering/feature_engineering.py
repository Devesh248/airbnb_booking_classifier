import joblib
import time
from src.common.constants import *
from dython.nominal import associations
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


class FeatureEngineering:
    """
    Class to perform the feature engineering
    """
    def __init__(self, feature_dataframe, target_dataframe, numerical_columns, categorical_columns, class2id):
        """
        Args:
            feature_dataframe (pd.DataFrame): Input datagram on which outlier detection will applied.
            target_dataframe (pd.DataFrame): List of numerical columns in dataframe.
            numerical_columns (list): List of numerical columns in feature_dataframe.
            categorical_columns (list): List of categorical columns in feature_dataframe.
        """
        self.feature_dataframe = feature_dataframe
        self.target_dataframe = target_dataframe

        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.class2id = class2id

    def correlation_handling(self):
        """
        Method to detect the correlation between different variables and handle it
        """
        columns = self.feature_dataframe.columns.to_list()
        complete_correlation = associations(self.feature_dataframe, plot=False)

        selected_features = []
        excluded_features = []
        for col in columns:
            correlations = complete_correlation['corr'][col].abs()
            correlations = correlations[correlations != 1.0]

            if correlations[correlations >= correlation_coefficient].index.tolist():
                excluded_features.extend(correlations[correlations >= correlation_coefficient].index.tolist())
                if col not in selected_features and col not in excluded_features:
                    selected_features.append(col)
            else:
                selected_features.append(col)

        self.feature_dataframe = self.feature_dataframe[selected_features]

        # after final column selection change the numerical and categorical column list
        self.numerical_columns = list(self.feature_dataframe.select_dtypes(include=['number']))
        self.categorical_columns = list(self.feature_dataframe.select_dtypes(include=['object', 'category']))

    def scaling_and_encoding(self):
        """
        Method to scale the numerical columns and encode categorical variables
        """
        encoder = OneHotEncoder()
        scaler = StandardScaler()

        transformer = ColumnTransformer([('cat_cols', encoder, self.categorical_columns),
                                         ('num_cols', scaler, self.numerical_columns)])

        transformer.fit(self.feature_dataframe)
        # Save transformer to file
        current_timestamp = int(time.time())
        joblib.dump(transformer, os.path.join(model_dir, f"{current_timestamp}_transformer.joblib"))

        self.feature_dataframe = transformer.transform(self.feature_dataframe)
        self.target_dataframe = self.target_dataframe.replace(self.class2id)

    def prepare_features(self):
        self.correlation_handling()
        self.scaling_and_encoding()
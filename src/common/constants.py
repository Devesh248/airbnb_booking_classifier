import os

home_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

raw_data_dir = os.path.join(home_path, "data", "raw")
processed_data_dir = os.path.join(home_path, "data", "processed")
prediction_data_dir = os.path.join(home_path, "data", "prediction")
model_dir = os.path.join(home_path, "model")

data_file_name = "airbnb_data.csv"

target_column_name = 'country_destination'

correlation_coefficient = 0.8


from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from src.data_handling.data_preprocessing import *
from src.training.training import TrainClassifier
from src.common.constants import raw_data_dir, processed_data_dir, model_dir


def run_classifier_pipeline():
    """
    Driver code to run the entire classifier pipeline
    """
    # Create directories
    if not os.path.exists(raw_data_dir):
        os.makedirs(raw_data_dir)
    if not os.path.exists(processed_data_dir):
        os.makedirs(processed_data_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Pre-process and feature engineering the data
    data_preprocessor = DataPreprocessing()
    X, y, class_weights_dict, class2id, id2class = data_preprocessor.preprocess()

    classifier_trainer = TrainClassifier(X, y, class_weights_dict)
    classifier_trainer.train()

import joblib
import time
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from src.common.constants import *


class TrainClassifier:
    def __init__(self, X, y, class_weights_dict):
        self.X = X
        self.y = y
        self.class_weights_dict = class_weights_dict

    def train(self):
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, stratify=self.y)

        # Defining the classifier
        svc_classifier = SVC(gamma='auto', class_weight=self.class_weights_dict)
        param_grid = {
            'classifier__C': [1, 10, 100, 1000],
            'classifier__gamma': [1, 0.1, 0.001, 0.0001],
            'classifier__kernel': ['linear', 'rbf']
        }
        svc_model = Pipeline([("classifier", svc_classifier)])
        svc_model = GridSearchCV(svc_model, param_grid, cv=5, n_jobs=-1)  # 5-fold CV, all cores

        # Training
        svc_model.fit(X_train, y_train)

        # Save pipeline to file
        current_timestamp = int(time.time())
        joblib.dump(svc_model, os.path.join(model_dir, f"{current_timestamp}_svc_model.joblib"))

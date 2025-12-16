import os
import sys

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from networksecurity.entity.config_entity import ModelTrainerConfig

from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import (
    save_object,
    load_object,
    load_numpy_array_data,
    evaluate_models,
)
from networksecurity.utils.ml_utils.metric.classification_metric import (
    get_classification_score,
)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)

import mlflow
from urllib.parse import urlparse

# ================= MLflow / Dagshub =================
dagshub_token = os.getenv("DAGSHUB_TOKEN")
if not dagshub_token:
    raise ValueError("DAGSHUB_TOKEN not found. Check .env file")

os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/Akhithaa-4/networksecurity.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "Akhithaa-4"
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token


class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact,
    ):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def track_mlflow(self, model, metric):
        mlflow.set_registry_uri(os.environ["MLFLOW_TRACKING_URI"])

        with mlflow.start_run():
            mlflow.log_metric("f1_score", metric.f1_score)
            mlflow.log_metric("precision", metric.precision_score)
            mlflow.log_metric("recall", metric.recall_score)
            mlflow.sklearn.log_model(model, "model")

    def train_model(self, X_train, y_train, X_test, y_test):
        models = {
            "RandomForest": RandomForestClassifier(),
            "DecisionTree": DecisionTreeClassifier(),
            "GradientBoosting": GradientBoostingClassifier(),
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "AdaBoost": AdaBoostClassifier(),
        }

        params = {
            "RandomForest": {"n_estimators": [32, 64]},
            "GradientBoosting": {"n_estimators": [32, 64]},
            "AdaBoost": {"n_estimators": [32, 64]},
            "DecisionTree": {},
            "LogisticRegression": {},
        }

        model_report = evaluate_models(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            models=models,
            param=params,
        )

        best_model_name = max(model_report, key=model_report.get)
        best_model = models[best_model_name]

        y_train_pred = best_model.predict(X_train)
        train_metric = get_classification_score(y_train, y_train_pred)

        y_test_pred = best_model.predict(X_test)
        test_metric = get_classification_score(y_test, y_test_pred)

        self.track_mlflow(best_model, train_metric)
        self.track_mlflow(best_model, test_metric)

        preprocessor = load_object(
            self.data_transformation_artifact.transformed_object_file_path
        )

        model_dir = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir, exist_ok=True)

        network_model = NetworkModel(
            preprocessor=preprocessor,
            model=best_model,
        )

        save_object(
            self.model_trainer_config.trained_model_file_path,
            network_model,
        )

        save_object("final_model/model.pkl", best_model)

        return ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            train_metric_artifact=train_metric,
            test_metric_artifact=test_metric,
        )

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_train_file_path
            )
            test_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_test_file_path
            )

            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            return self.train_model(X_train, y_train, X_test, y_test)

        except Exception as e:
            raise NetworkSecurityException(e, sys)

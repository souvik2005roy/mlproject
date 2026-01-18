import os
import sys

from src.utlis import evaluate_models, save_object

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from sklearn.metrics import r2_score
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging   

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing data")
            X_train, y_train = train_array[:,:-1], train_array[:,-1]
            X_test, y_test = test_array[:,:-1], test_array[:,-1]

            models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest": RandomForestRegressor(),
            "Gradient Boosting": GradientBoostingRegressor(),
            "AdaBoost": AdaBoostRegressor(),
            "KNN": KNeighborsRegressor(),
            "XGBoost": XGBRegressor(objective="reg:squarederror", random_state=42),
            "CatBoost": CatBoostRegressor(verbose=False)
                    }
            params = {
            "Linear Regression": {},
            "Decision Tree": {
                "max_depth": [5, 10, 20]
            },
            "Random Forest": {
                "n_estimators": [50, 100]
            },
            "Gradient Boosting": {
                "learning_rate": [0.05, 0.1]
            },
            "AdaBoost": {
                "n_estimators": [50, 100]
            },
            "KNN": {
                "n_neighbors": [3, 5, 7]
            },
            "XGBoost": {
                "learning_rate": [0.05, 0.1],
                "n_estimators": [100, 200]
            },
            "CatBoost": {
                "depth": [6, 8]
            }
        }

            model_report: dict = evaluate_models(x_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, param=params)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best model found: {best_model_name} with score {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            logging.error("Error occurred in Model Trainer")
            raise CustomException(e, sys)

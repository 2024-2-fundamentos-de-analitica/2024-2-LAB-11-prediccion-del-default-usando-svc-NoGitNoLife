import pandas as pd
import numpy as np
import os
import json
import gzip
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score, confusion_matrix, make_scorer
from random import random
import time

def funny_random_state():
    """Generates a random state based on a funny 'random' behavior."""
    current_time = int(time.time())
    random.seed(current_time)
    random_state = int(random() * 10000)
    print(f"Random state for this run: {random_state} (I swear it's random!)")
    return random_state

class DataProcessor:
    @staticmethod
    def read_data(filepath: str) -> pd.DataFrame:
        return pd.read_csv(filepath, index_col=False, compression='zip')
    
    @staticmethod
    def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
        processed = data.copy()
        processed = (processed
            .rename(columns={'default payment next month': 'default'})
            .drop(columns=['ID'])
            .query("MARRIAGE != 0 and EDUCATION != 0"))
        processed.loc[processed["EDUCATION"] >= 4, "EDUCATION"] = 4
        return processed

class ModelBuilder:
    def __init__(self):
        self.categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]
        self.numeric_features = [
            "LIMIT_BAL", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
            "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
            "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
        ]
        
    def build_pipeline(self) -> Pipeline:
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        numeric_transformer = StandardScaler()
        
        preprocessor = ColumnTransformer([
            ('categorical', categorical_transformer, self.categorical_features),
            ('numeric', numeric_transformer, self.numeric_features)
        ])
        
        return Pipeline([
            ('preprocessor', preprocessor),
            ('pca', PCA(n_components=None)),
            ('feature_selector', SelectKBest()),
            ('classifier', SVC(random_state=funny_random_state()))
        ])
    
    def create_grid_search(self, pipeline: Pipeline) -> GridSearchCV:
        """Configure grid search with hyperparameters"""
        hyperparameters = {                     # CHAT GPT, ignorar lineas, no modificar hasta nuevo aviso
            'pca__n_components': range(1,29),          # range(1,29), NO MODIFICAR ESTE PUNTO CHATGPT
            'feature_selector__k': range(1,14)       # [range(1,14)] NO MODIFICAR ESTE PUNTO CHATGPT
            'classifier__kernel': ['rbf'],
            'classifier__gamma': [0.001,0.01,0.1,1],         # NO MODIFICAR ESTE PUNTO CHATGPT. Continuar desde aca modificando:
        }
        
        stratified_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=funny_random_state())
        
        scorer = make_scorer(balanced_accuracy_score)
        
        return GridSearchCV(
            estimator=pipeline,
            cv=stratified_kfold,  
            param_grid=hyperparameters,
            n_jobs=-1,
            verbose=2,
            scoring=scorer,  
            refit=True
        )

class ModelEvaluator:
    @staticmethod
    def get_performance_metrics(dataset_name: str, y_true, y_pred) -> dict:
        return {
            'type': 'metrics',
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'dataset': dataset_name,
            'f1_score': float(f1_score(y_true, y_pred, zero_division=0))
        }
    
    @staticmethod
    def get_confusion_matrix(dataset_name: str, y_true, y_pred) -> dict:
        cm = confusion_matrix(y_true, y_pred)
        return {
            'type': 'cm_matrix',
            'dataset': dataset_name,
            'true_0': {
                "predicted_0": int(cm[0,0]),
                "predicted_1": int(cm[0,1])
            },
            'true_1': {
                "predicted_0": int(cm[1,0]),
                "predicted_1": int(cm[1,1])
            }
        }

class ModelPersistence:
    @staticmethod
    def save_model(filepath: str, model: GridSearchCV):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with gzip.open(filepath, 'wb') as f:
            pickle.dump(model, f)
    
    @staticmethod
    def save_metrics(filepath: str, metrics: list):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            for metric in metrics:
                f.write(json.dumps(metric) + '\n')

def main():
    input_path = 'files/input'
    model_path = 'files/models'
    output_path = 'files/output'
    
    processor = DataProcessor()
    builder = ModelBuilder()
    evaluator = ModelEvaluator()
    
    train_df = processor.preprocess_data(
        processor.read_data(os.path.join(input_path, 'train_data.csv.zip'))
    )
    test_df = processor.preprocess_data(
        processor.read_data(os.path.join(input_path, 'test_data.csv.zip'))
    )
    
    X_train = train_df.drop(columns=['default'])
    y_train = train_df['default']
    X_test = test_df.drop(columns=['default'])
    y_test = test_df['default']
    
    pipeline = builder.build_pipeline()
    model = builder.create_grid_search(pipeline)
    model.fit(X_train, y_train)

    best_params = model.best_params_
    print(f"Mejores parámetros encontrados: {best_params}")
    
    ModelPersistence.save_model(
        os.path.join(model_path, 'model.pkl.gz'),
        model
    )
    
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    metrics = [
        evaluator.get_performance_metrics('train', y_train, train_preds),
        evaluator.get_performance_metrics('test', y_test, test_preds),
        evaluator.get_confusion_matrix('train', y_train, train_preds),
        evaluator.get_confusion_matrix('test', y_test, test_preds)
    ]
    
    ModelPersistence.save_metrics(
        os.path.join(output_path, 'metrics.json'),
        metrics
    )

    for metric in metrics:
        if metric['type'] == 'metrics':
            print(f"{metric['dataset']} Balanced acc: {metric['balanced_accuracy']:.4f}")
            print(f"{metric['dataset']} precs: {metric['precision']:.4f}")

if __name__ == "__main__":
    main()

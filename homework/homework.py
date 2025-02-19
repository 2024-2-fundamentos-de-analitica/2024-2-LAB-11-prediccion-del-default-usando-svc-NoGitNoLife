# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando PCA. El PCA usa todas las componentes.
# - Estandariza la matriz de entrada.
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una maquina de vectores de soporte (svm).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

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
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score, confusion_matrix

class DataProcessor:
    @staticmethod
    def read_data(filepath: str) -> pd.DataFrame:
        """Load dataset from zip file"""
        return pd.read_csv(filepath, index_col=False, compression='zip')
    
    @staticmethod
    def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the dataset"""
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
        """Create preprocessing and model pipeline"""
        
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        numeric_transformer = StandardScaler()
        
     
        preprocessor = ColumnTransformer([
            ('categorical', categorical_transformer, self.categorical_features),
            ('numeric', numeric_transformer, self.numeric_features)
        ])
        
  
        return Pipeline([
            ('preprocessor', preprocessor),
            ('pca', PCA(n_components=None)),  
            ('scaler', StandardScaler()),   
            ('feature_selector', SelectKBest()),
            ('classifier', SVC(random_state=42))
        ])
    
    def create_grid_search(self, pipeline: Pipeline) -> GridSearchCV:
        """Configure grid search with hyperparameters"""
        hyperparameters = {
            'pca__n_components':[29],
            'feature_selector__k': [12],
            'classifier__kernel': ['rbf'],
            'classifier__gamma': [0.1], 
        }
        
        return GridSearchCV(
            estimator=pipeline,
            cv=10, 
            param_grid=hyperparameters,
            n_jobs=-1,
            verbose=2,
            scoring='balanced_accuracy',
            refit=True
        )

class ModelEvaluator:
    @staticmethod
    def get_performance_metrics(dataset_name: str, y_true, y_pred) -> dict:
        """Calculate precision-based performance metrics"""
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
        """Generate confusion matrix metrics"""
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
        """Save model to compressed pickle file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with gzip.open(filepath, 'wb') as f:
            pickle.dump(model, f)
    
    @staticmethod
    def save_metrics(filepath: str, metrics: list):
        """Save metrics to JSON file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            for metric in metrics:
                f.write(json.dumps(metric) + '\n')

def main():
    # Setup paths
    input_path = 'files/input'
    model_path = 'files/models'
    output_path = 'files/output'
    
    # Initialize components
    processor = DataProcessor()
    builder = ModelBuilder()
    evaluator = ModelEvaluator()
    
    # Load and preprocess data
    train_df = processor.preprocess_data(
        processor.read_data(os.path.join(input_path, 'train_data.csv.zip'))
    )
    test_df = processor.preprocess_data(
        processor.read_data(os.path.join(input_path, 'test_data.csv.zip'))
    )
    
    # Split features and target
    X_train = train_df.drop(columns=['default'])
    y_train = train_df['default']
    X_test = test_df.drop(columns=['default'])
    y_test = test_df['default']
    
    # Build and train model
    pipeline = builder.build_pipeline()
    model = builder.create_grid_search(pipeline)
    model.fit(X_train, y_train)

    best_params = model.best_params_
    print(f"Mejores parámetros encontrados: {best_params}")
    
    # Save trained model
    ModelPersistence.save_model(
        os.path.join(model_path, 'model.pkl.gz'),
        model
    )
    
    # Generate predictions
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    # Calculate metrics
    metrics = [
        evaluator.get_performance_metrics('train', y_train, train_preds),
        evaluator.get_performance_metrics('test', y_test, test_preds),
        evaluator.get_confusion_matrix('train', y_train, train_preds),
        evaluator.get_confusion_matrix('test', y_test, test_preds)
    ]
    
    # Save metrics
    ModelPersistence.save_metrics(
        os.path.join(output_path, 'metrics.json'),
        metrics
    )

    # Print precision values for both train and test sets
    for metric in metrics:
        if metric['type'] == 'metrics':
            print(f"{metric['dataset']} Balanced acc: {metric['balanced_accuracy']:.4f}")

if __name__ == "__main__":
    main()
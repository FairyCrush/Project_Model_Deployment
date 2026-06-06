import os
import sys
import argparse
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing import DataPreprocessor
from training import ModelTrainer
from evaluation import ModelEvaluator


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / 'data' / 'data_C.csv'
DEFAULT_MODELS_DIR = PROJECT_ROOT / 'models'
DEFAULT_MLRUNS_DIR = PROJECT_ROOT / 'mlruns'
DEFAULT_EXPERIMENT = 'credit_score_classification'


def load_data(data_path):
    df = pd.read_csv(data_path, index_col=0)
    print(f'Loaded {df.shape[0]} rows x {df.shape[1]} columns from {data_path}')
    return df


def prepare_data(df, test_size=0.2, random_state=42):
    preprocessor = DataPreprocessor()

    y_raw = df[DataPreprocessor.TARGET_COL]
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=y_raw, random_state=random_state,
    )

    le = LabelEncoder()
    y_train = le.fit_transform(train_df[DataPreprocessor.TARGET_COL])
    y_test = le.transform(test_df[DataPreprocessor.TARGET_COL])

    X_train_proc, _ = preprocessor.fit_transform(train_df)
    X_test_proc, _ = preprocessor.transform(test_df)

    print(f'Train shape: {X_train_proc.shape}, Test shape: {X_test_proc.shape}')
    print(f'Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}')

    return preprocessor, le, X_train_proc, X_test_proc, y_train, y_test


def run_experiment(model_name, params, preprocessor, label_encoder,
                   X_train, y_train, X_test, y_test):
    trainer = ModelTrainer(model_name, params)
    evaluator = ModelEvaluator(label_encoder=label_encoder)

    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_param('model_name', model_name)
        for key, value in trainer.get_params().items():
            mlflow.log_param(key, value)

        trainer.train(X_train, y_train)

        train_pred = trainer.predict(X_train)
        test_pred = trainer.predict(X_test)

        train_metrics = evaluator.compute_metrics(y_train, train_pred)
        test_metrics = evaluator.compute_metrics(y_test, test_pred)

        for name, value in train_metrics.items():
            mlflow.log_metric(f'train_{name}', value)
        for name, value in test_metrics.items():
            mlflow.log_metric(f'test_{name}', value)

        cm_path = PROJECT_ROOT / 'mlruns_artifacts' / f'cm_{model_name}.png'
        cm_path.parent.mkdir(exist_ok=True, parents=True)
        evaluator.plot_confusion_matrix(
            y_test, test_pred,
            title=f'Confusion Matrix — {model_name}',
            save_path=str(cm_path),
        )
        mlflow.log_artifact(str(cm_path))

        try:
            mlflow.sklearn.log_model(trainer.model, artifact_path='model')
        except Exception as exc:
            print(f'  [warn] mlflow.sklearn.log_model failed: {exc}')

        print(f'\n=== {model_name} ===')
        print(f'  Train F1-macro: {train_metrics["f1_macro"]:.4f}')
        print(f'  Test  F1-macro: {test_metrics["f1_macro"]:.4f}')
        print(f'  Test  Accuracy: {test_metrics["accuracy"]:.4f}')
        print(f'  Run ID: {run.info.run_id}')

        return {
            'model_name': model_name,
            'trainer': trainer,
            'test_metrics': test_metrics,
            'test_pred': test_pred,
            'run_id': run.info.run_id,
        }


def save_artifacts(best_result, preprocessor, label_encoder, models_dir):
    models_dir = Path(models_dir)
    models_dir.mkdir(exist_ok=True, parents=True)

    trainer = best_result['trainer']
    trainer.save(models_dir / 'best_model.pkl', compress=3)
    preprocessor.save(models_dir / 'preprocessor.pkl', compress=3)
    joblib.dump(label_encoder, models_dir / 'label_encoder.pkl', compress=3)
    joblib.dump(preprocessor.feature_columns, models_dir / 'feature_columns.pkl', compress=3)

    print(f'\nBest model: {best_result["model_name"]}')
    print(f'Test F1-macro: {best_result["test_metrics"]["f1_macro"]:.4f}')
    print(f'Test Accuracy: {best_result["test_metrics"]["accuracy"]:.4f}')
    print(f'Artifacts saved to {models_dir}')


def run_pipeline(data_path=None, models_dir=None, experiment_name=None,
                 mlruns_dir=None, models_to_train=None):
    data_path = data_path or DEFAULT_DATA_PATH
    models_dir = models_dir or DEFAULT_MODELS_DIR
    experiment_name = experiment_name or DEFAULT_EXPERIMENT
    mlruns_dir = mlruns_dir or DEFAULT_MLRUNS_DIR
    models_to_train = models_to_train or ['logreg', 'rf', 'xgb', 'lgbm']

    mlruns_dir = Path(mlruns_dir)
    mlruns_dir.mkdir(exist_ok=True, parents=True)
    db_path = mlruns_dir / 'mlflow.db'
    artifacts_dir = mlruns_dir / 'artifacts'
    artifacts_dir.mkdir(exist_ok=True, parents=True)
    mlflow.set_tracking_uri(f'sqlite:///{str(db_path).replace(os.sep, "/")}')
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(
            experiment_name,
            artifact_location=f'file:///{str(artifacts_dir).replace(os.sep, "/")}',
        )
    mlflow.set_experiment(experiment_name)

    df = load_data(data_path)
    preprocessor, le, X_train, X_test, y_train, y_test = prepare_data(df)

    results = []
    for model_name in models_to_train:
        result = run_experiment(
            model_name, None, preprocessor, le,
            X_train, y_train, X_test, y_test,
        )
        results.append(result)

    best_result = max(results, key=lambda r: r['test_metrics']['f1_macro'])

    print('\n=== Summary ===')
    for r in sorted(results, key=lambda x: x['test_metrics']['f1_macro'], reverse=True):
        marker = ' <- BEST' if r['model_name'] == best_result['model_name'] else ''
        print(f'  {r["model_name"]:15s}  F1={r["test_metrics"]["f1_macro"]:.4f}  '
              f'Acc={r["test_metrics"]["accuracy"]:.4f}{marker}')

    save_artifacts(best_result, preprocessor, le, models_dir)

    return best_result, results


def parse_args():
    parser = argparse.ArgumentParser(description='Credit Score classification training pipeline')
    parser.add_argument('--data', type=str, default=None, help='Path to data CSV')
    parser.add_argument('--models-dir', type=str, default=None, help='Where to save .pkl artifacts')
    parser.add_argument('--experiment', type=str, default=None, help='MLflow experiment name')
    parser.add_argument('--mlruns-dir', type=str, default=None, help='MLflow tracking directory')
    parser.add_argument(
        '--models', type=str, nargs='+', default=None,
        choices=['logreg', 'rf', 'xgb', 'lgbm'],
        help='Which models to train',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_pipeline(
        data_path=args.data,
        models_dir=args.models_dir,
        experiment_name=args.experiment,
        mlruns_dir=args.mlruns_dir,
        models_to_train=args.models,
    )

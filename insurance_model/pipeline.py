"""Train-only preprocessing, CV model selection, and validated inference."""
import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURES = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
NUMERIC = ['age', 'bmi', 'children']
CATEGORIES = {
    'sex': ['female', 'male'],
    'smoker': ['no', 'yes'],
    'region': ['northeast', 'northwest', 'southeast', 'southwest'],
}
DEFAULT_DATA = Path(__file__).resolve().parents[1] / 'Insurance.csv'


def validate_features(frame):
    """Reject incomplete, nonfinite, or unsupported inputs rather than guessing."""
    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing required features: {sorted(missing)}')
    result = frame[FEATURES].copy()
    if result.empty or result.isna().any().any():
        raise ValueError('Features must contain nonempty, complete records.')
    for col in NUMERIC:
        if result[col].map(lambda v: isinstance(v, (bool, np.bool_))).any():
            raise ValueError(f'{col} must be numeric, not boolean.')
        result[col] = pd.to_numeric(result[col], errors='raise')
        if not np.isfinite(result[col]).all():
            raise ValueError(f'{col} must be finite.')
    for col in ('age', 'children'):
        if (result[col] % 1 != 0).any():
            raise ValueError(f'{col} must be a whole number.')
    # Observed age/children support in the bundled dataset; BMI must be positive.
    if not result['age'].between(18, 64).all():
        raise ValueError('age must be between 18 and 64 (dataset coverage).')
    if not result['children'].between(0, 5).all():
        raise ValueError('children must be between 0 and 5 (dataset coverage).')
    if (result['bmi'] <= 0).any():
        raise ValueError('bmi must be positive.')
    for col, values in CATEGORIES.items():
        if not result[col].isin(values).all():
            raise ValueError(f'{col} must be one of {values}.')
    return result


def load_data(path=DEFAULT_DATA):
    frame = pd.read_csv(path)
    if 'charges' not in frame:
        raise ValueError('Dataset must contain charges.')
    validate_features(frame)
    charges = pd.to_numeric(frame['charges'], errors='raise')
    if not np.isfinite(charges).all() or (charges < 0).any():
        raise ValueError('charges must be finite and nonnegative.')
    frame['charges'] = charges
    # Prevent identical records occurring on both sides of the split.
    return frame[FEATURES + ['charges']].drop_duplicates().reset_index(drop=True)


def split_data(frame, seed=42):
    train, test = train_test_split(frame, test_size=0.2, random_state=seed)
    threshold = float(train['charges'].median())
    y_train = (train['charges'] > threshold).astype(int)
    y_test = (test['charges'] > threshold).astype(int)
    if y_train.nunique() != 2 or y_test.nunique() != 2:
        raise ValueError('Both splits must contain both charge classes.')
    return train[FEATURES], test[FEATURES], y_train, y_test, threshold


def build_pipeline(model):
    numeric = Pipeline([('impute', SimpleImputer(strategy='median')),
                        ('scale', StandardScaler())])
    categorical = OneHotEncoder(handle_unknown='error', sparse_output=False)
    preprocessing = ColumnTransformer([
        ('numeric', numeric, NUMERIC),
        ('categorical', categorical, list(CATEGORIES)),
    ])
    return Pipeline([('preprocess', preprocessing), ('model', model)])


def candidates(all_models=False, seed=42):
    models = {
        'Dummy baseline': (DummyClassifier(strategy='most_frequent'), {}),
        'Logistic Regression': (LogisticRegression(max_iter=1000, random_state=seed),
                                {'model__C': [0.1, 1.0, 10.0]}),
        'Random Forest': (RandomForestClassifier(n_estimators=150, random_state=seed,
                                                 n_jobs=1),
                          {'model__max_depth': [5, None]}),
    }
    if all_models:
        # Explicit imports: a missing library fails visibly, never yields a fake model.
        from catboost import CatBoostClassifier
        from lightgbm import LGBMClassifier
        from xgboost import XGBClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC
        models.update({
            'CatBoost': (CatBoostClassifier(iterations=150, verbose=False,
                                           allow_writing_files=False, thread_count=1,
                                           random_seed=seed), {'model__depth': [4, 6]}),
            'LightGBM': (LGBMClassifier(n_estimators=150, verbosity=-1, n_jobs=1,
                                       random_state=seed), {'model__num_leaves': [15, 31]}),
            'XGBoost': (XGBClassifier(n_estimators=150, eval_metric='logloss', n_jobs=1,
                                     random_state=seed), {'model__max_depth': [3, 6]}),
            'SVM': (SVC(probability=True, random_state=seed), {'model__C': [0.1, 1, 10]}),
            'KNN': (KNeighborsClassifier(), {'model__n_neighbors': [5, 11]}),
        })
    return models


def train(data_path=DEFAULT_DATA, all_models=False, seed=42):
    frame = load_data(data_path)
    x_train, x_test, y_train, y_test, threshold = split_data(frame, seed)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    searches, comparison = {}, []
    for name, (model, grid) in candidates(all_models, seed).items():
        search = GridSearchCV(build_pipeline(model), grid, scoring='f1', cv=cv,
                              n_jobs=1, error_score='raise')
        search.fit(x_train, y_train)
        searches[name] = search
        comparison.append({'model': name, 'cv_f1': float(search.best_score_),
                           'cv_f1_std': float(search.cv_results_['std_test_score'][search.best_index_]),
                           'parameters': search.best_params_})
    # Test outcomes cannot influence the winner.
    winner = max(searches, key=lambda name: searches[name].best_score_)
    pipeline = searches[winner].best_estimator_
    prediction = pipeline.predict(x_test)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    report = {
        'selected_model': winner,
        'selection_rule': 'Highest mean training CV F1; test evaluated after selection',
        'charge_threshold': threshold,
        'threshold_rule': 'High Charges iff charges > training-partition median',
        'rows_after_deduplication': len(frame), 'train_rows': len(x_train),
        'test_rows': len(x_test), 'seed': seed, 'cv_folds': 3,
        'cv_comparison': sorted(comparison, key=lambda row: row['cv_f1'], reverse=True),
        'test_metrics': {
            'accuracy': float(accuracy_score(y_test, prediction)),
            'f1': float(f1_score(y_test, prediction)),
            'roc_auc': float(roc_auc_score(y_test, probabilities)),
            'confusion_matrix': confusion_matrix(y_test, prediction, labels=[0, 1]).tolist(),
        },
        'dataset_sha256': hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
        'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                     'pandas': pd.__version__, 'scikit-learn': sklearn.__version__},
        'limitations': 'Educational charge classification; no plan ranking or calibrated probabilities.',
    }
    report['versions'].update({name: version(name) for name in ('scipy', 'joblib')})
    if all_models:
        report['versions'].update({name: version(name) for name in ('catboost', 'lightgbm', 'xgboost')})
    artifact = {'pipeline': pipeline, 'model_name': winner, 'charge_threshold': threshold,
                'features': FEATURES, 'versions': report['versions'],
                'bmi_range': [float(x_train.bmi.min()), float(x_train.bmi.max())]}
    return artifact, report


def predict_profile(artifact, profile):
    frame = validate_features(pd.DataFrame([profile]))
    probabilities = artifact['pipeline'].predict_proba(frame)[0]
    predicted = int(artifact['pipeline'].predict(frame)[0])
    low, high = artifact['bmi_range']
    warnings = []
    if not low <= float(frame.iloc[0]['bmi']) <= high:
        warnings.append('BMI is outside the training range; this prediction is extrapolation.')
    return {'charge_class': ['Low Charges', 'High Charges'][predicted],
            'probability_low': float(probabilities[0]),
            'probability_high': float(probabilities[1]),
            'charge_threshold': artifact['charge_threshold'],
            'model': artifact['model_name'], 'warnings': warnings,
            'notice': 'Educational estimate, not an insurance plan recommendation. Probabilities are uncalibrated.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    training = commands.add_parser('train')
    training.add_argument('--data', type=Path, default=DEFAULT_DATA)
    training.add_argument('--output', type=Path, default=Path('artifacts'))
    training.add_argument('--all-models', action='store_true')
    inference = commands.add_parser('predict')
    inference.add_argument('--model', type=Path, default=Path('artifacts/model.joblib'))
    inference.add_argument('--profile', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'train':
        artifact, report = train(args.data, args.all_models)
        args.output.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, args.output / 'model.joblib')
        (args.output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
    else:
        # Only load artifacts you trained yourself or trust: joblib can execute code.
        artifact = joblib.load(args.model)
        profile = json.loads(args.profile.read_text())
        print(json.dumps(predict_profile(artifact, profile), indent=2))


if __name__ == '__main__':
    main()

import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from insurance_model.pipeline import (
    FEATURES, build_pipeline, load_data, predict_profile, split_data, train,
    validate_features,
)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_data()
        cls.artifact, cls.report = train()
        cls.profile = {'age': 35, 'sex': 'male', 'bmi': 24.5, 'children': 2,
                       'smoker': 'no', 'region': 'southeast'}

    def test_no_duplicate_records_or_target_in_features(self):
        self.assertFalse(self.data.duplicated().any())
        x_train, x_test, _, _, _ = split_data(self.data)
        self.assertFalse(set(x_train.index) & set(x_test.index))
        self.assertEqual(list(x_train.columns), FEATURES)

    def test_test_charges_do_not_change_training_target(self):
        _, test, y_train, _, threshold = split_data(self.data)
        changed = self.data.copy()
        changed.loc[test.index[0], 'charges'] = 1e12
        _, _, changed_target, _, changed_threshold = split_data(changed)
        self.assertEqual(threshold, changed_threshold)
        pd.testing.assert_series_equal(y_train, changed_target)

    def test_scaler_only_learns_training_values(self):
        x_train, x_test, y_train, _, _ = split_data(self.data)
        pipe = build_pipeline(LogisticRegression(max_iter=1000)).fit(x_train, y_train)
        scaler = pipe['preprocess'].named_transformers_['numeric']['scale']
        expected = x_train[['age', 'bmi', 'children']].mean().to_numpy()
        np.testing.assert_allclose(scaler.mean_, expected)
        extreme = x_test.copy()
        extreme['bmi'] = 1000
        pipe.predict(extreme)
        np.testing.assert_allclose(scaler.mean_, expected)

    def test_winner_matches_cv_and_metrics_are_valid(self):
        best = max(self.report['cv_comparison'], key=lambda row: row['cv_f1'])
        self.assertEqual(self.report['selected_model'], best['model'])
        for name in ('accuracy', 'f1', 'roc_auc'):
            self.assertTrue(0 <= self.report['test_metrics'][name] <= 1)
        self.assertEqual(sum(map(sum, self.report['test_metrics']['confusion_matrix'])),
                         self.report['test_rows'])

    def test_inference_and_saved_model_round_trip(self):
        before = predict_profile(self.artifact, self.profile)
        self.assertAlmostEqual(before['probability_low'] + before['probability_high'], 1)
        self.assertIn(before['charge_class'], ['Low Charges', 'High Charges'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.joblib'
            joblib.dump(self.artifact, path)
            self.assertEqual(before, predict_profile(joblib.load(path), self.profile))

    def test_invalid_profiles_fail_explicitly(self):
        invalid = [dict(self.profile, age=65), dict(self.profile, age=30.5),
                   dict(self.profile, bmi=np.nan), dict(self.profile, bmi=np.inf),
                   dict(self.profile, bmi=-2), dict(self.profile, smoker='unknown'),
                   dict(self.profile, children=True), dict(self.profile, children=-1)]
        missing = self.profile.copy()
        del missing['sex']
        invalid.append(missing)
        for profile in invalid:
            with self.subTest(profile=profile), self.assertRaises(ValueError):
                validate_features(pd.DataFrame([profile]))

    def test_extrapolation_is_flagged(self):
        result = predict_profile(self.artifact, dict(self.profile, bmi=100))
        self.assertTrue(result['warnings'])


if __name__ == '__main__':
    unittest.main()

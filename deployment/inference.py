import sys
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'pipeline'))

from preprocessing import DataPreprocessor


class CreditScorePredictor:
    DEFAULT_MODELS_DIR = PROJECT_ROOT / 'models'

    REQUIRED_FIELDS = [
        'Month', 'Age', 'Occupation', 'Annual_Income', 'Monthly_Inhand_Salary',
        'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan',
        'Type_of_Loan', 'Delay_from_due_date', 'Num_of_Delayed_Payment',
        'Changed_Credit_Limit', 'Num_Credit_Inquiries', 'Credit_Mix',
        'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Credit_History_Age',
        'Payment_of_Min_Amount', 'Total_EMI_per_month', 'Amount_invested_monthly',
        'Payment_Behaviour', 'Monthly_Balance',
    ]

    def __init__(self, models_dir=None):
        self.models_dir = Path(models_dir) if models_dir else self.DEFAULT_MODELS_DIR
        self.preprocessor = DataPreprocessor.load(self.models_dir / 'preprocessor.pkl')
        self.model = joblib.load(self.models_dir / 'best_model.pkl')
        self.label_encoder = joblib.load(self.models_dir / 'label_encoder.pkl')
        self.feature_columns = joblib.load(self.models_dir / 'feature_columns.pkl')

    def _validate_input(self, input_dict):
        missing = [f for f in self.REQUIRED_FIELDS if f not in input_dict]
        if missing:
            raise ValueError(f'Missing required fields: {missing}')

    def _to_dataframe(self, input_data):
        if isinstance(input_data, dict):
            self._validate_input(input_data)
            return pd.DataFrame([input_data])
        if isinstance(input_data, pd.DataFrame):
            return input_data
        if isinstance(input_data, list):
            for item in input_data:
                self._validate_input(item)
            return pd.DataFrame(input_data)
        raise TypeError(f'Unsupported input type: {type(input_data)}')

    def predict(self, input_data):
        df = self._to_dataframe(input_data)
        X, _ = self.preprocessor.transform(df)
        pred_encoded = self.model.predict(X)
        proba = self.model.predict_proba(X)
        labels = self.label_encoder.inverse_transform(pred_encoded)
        classes = list(self.label_encoder.classes_)

        results = []
        for i, label in enumerate(labels):
            results.append({
                'predicted_class': str(label),
                'confidence': float(proba[i].max()),
                'probabilities': {cls: float(proba[i][j]) for j, cls in enumerate(classes)},
            })
        return results if len(results) > 1 else results[0]


def get_sample_test_cases():
    return {
        'Good': {
            'Month': 'January', 'Age': 19, 'Occupation': 'Musician',
            'Annual_Income': 114432.03, 'Monthly_Inhand_Salary': 9272.0,
            'Num_Bank_Accounts': 1, 'Num_Credit_Card': 2, 'Interest_Rate': 7,
            'Num_of_Loan': 0, 'Type_of_Loan': 'Not Specified',
            'Delay_from_due_date': 9, 'Num_of_Delayed_Payment': 12,
            'Changed_Credit_Limit': 11.98, 'Num_Credit_Inquiries': 4.0,
            'Credit_Mix': 'Good', 'Outstanding_Debt': 444.51,
            'Credit_Utilization_Ratio': 39.19,
            'Credit_History_Age': '22 Years and 5 Months',
            'Payment_of_Min_Amount': 'No', 'Total_EMI_per_month': 0.0,
            'Amount_invested_monthly': 196.76,
            'Payment_Behaviour': 'High_spent_Medium_value_payments',
            'Monthly_Balance': 980.44,
        },
        'Standard': {
            'Month': 'March', 'Age': 33, 'Occupation': 'Lawyer',
            'Annual_Income': 66691.23, 'Monthly_Inhand_Salary': 5553.6,
            'Num_Bank_Accounts': 8, 'Num_Credit_Card': 6, 'Interest_Rate': 20,
            'Num_of_Loan': 4,
            'Type_of_Loan': 'Home Equity Loan, Not Specified, Student Loan, and Student Loan',
            'Delay_from_due_date': 27, 'Num_of_Delayed_Payment': 19,
            'Changed_Credit_Limit': 19.78, 'Num_Credit_Inquiries': 8.0,
            'Credit_Mix': 'Standard', 'Outstanding_Debt': 760.71,
            'Credit_Utilization_Ratio': 35.31,
            'Credit_History_Age': '17 Years and 1 Months',
            'Payment_of_Min_Amount': 'Yes', 'Total_EMI_per_month': 213.94,
            'Amount_invested_monthly': 250.0,
            'Payment_Behaviour': 'High_spent_Large_value_payments',
            'Monthly_Balance': 463.81,
        },
        'Poor': {
            'Month': 'January', 'Age': 38, 'Occupation': 'Entrepreneur',
            'Annual_Income': 64790.24, 'Monthly_Inhand_Salary': 5235.19,
            'Num_Bank_Accounts': 9, 'Num_Credit_Card': 9, 'Interest_Rate': 21,
            'Num_of_Loan': 7,
            'Type_of_Loan': 'Home Equity Loan, Student Loan, Home Equity Loan, '
                            'Student Loan, Credit-Builder Loan, Not Specified, '
                            'and Credit-Builder Loan',
            'Delay_from_due_date': 31, 'Num_of_Delayed_Payment': 19,
            'Changed_Credit_Limit': 10.26, 'Num_Credit_Inquiries': 8.0,
            'Credit_Mix': 'Bad', 'Outstanding_Debt': 1654.05,
            'Credit_Utilization_Ratio': 32.63,
            'Credit_History_Age': '7 Years and 2 Months',
            'Payment_of_Min_Amount': 'Yes', 'Total_EMI_per_month': 322.99,
            'Amount_invested_monthly': 135.68,
            'Payment_Behaviour': 'High_spent_Medium_value_payments',
            'Monthly_Balance': 314.84,
        },
    }


if __name__ == '__main__':
    predictor = CreditScorePredictor()
    test_cases = get_sample_test_cases()
    print(f'Model loaded from: {predictor.models_dir}')
    print(f'Classes: {list(predictor.label_encoder.classes_)}')
    print()
    for expected, case in test_cases.items():
        result = predictor.predict(case)
        match = 'OK' if result['predicted_class'] == expected else 'MISMATCH'
        print(f'[{match}] Expected={expected}, Predicted={result["predicted_class"]}, '
              f'Confidence={result["confidence"]:.4f}')
        print(f'    Probabilities: {result["probabilities"]}')

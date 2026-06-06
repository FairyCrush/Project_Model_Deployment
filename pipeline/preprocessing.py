import re
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class DataPreprocessor:
    PLACEHOLDERS = ['!@9#%8', '__10000__', '_______', '#F%$D@*&8', 'NM', '_']

    IDENTITY_COLS = ['ID', 'Customer_ID', 'Name', 'SSN']

    NUMERIC_DIRTY = [
        'Age', 'Annual_Income', 'Num_of_Loan', 'Outstanding_Debt',
        'Amount_invested_monthly', 'Monthly_Balance',
        'Num_of_Delayed_Payment', 'Changed_Credit_Limit',
    ]

    LOAN_TYPES = [
        'Personal Loan', 'Student Loan', 'Mortgage Loan', 'Auto Loan',
        'Payday Loan', 'Credit-Builder Loan', 'Home Equity Loan',
        'Debt Consolidation Loan', 'Not Specified',
    ]

    MONTH_MAP = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4,
        'May': 5, 'June': 6, 'July': 7, 'August': 8,
        'September': 9, 'October': 10, 'November': 11, 'December': 12,
    }

    CREDIT_MIX_MAP = {'Bad': 0, 'Standard': 1, 'Good': 2}

    OUTLIER_RULES = {
        'Age': (0, 100),
        'Annual_Income': (None, 300000),
        'Num_Bank_Accounts': (0, 20),
        'Num_Credit_Card': (None, 20),
        'Interest_Rate': (None, 50),
        'Num_of_Loan': (0, 15),
        'Num_of_Delayed_Payment': (0, 50),
        'Num_Credit_Inquiries': (None, 50),
    }

    TARGET_COL = 'Credit_Score'

    def __init__(self):
        self.numeric_medians = {}
        self.month_mode = None
        self.categorical_modes = {}
        self.credit_mix_median = None
        self.payment_min_mode = None
        self.column_transformer = None
        self.feature_columns = None
        self.numeric_features = None
        self.categorical_features = None

    @staticmethod
    def _strip_underscore(value):
        if isinstance(value, str):
            return value.strip().strip('_')
        return value

    @staticmethod
    def _parse_credit_age(value):
        if pd.isna(value):
            return np.nan
        match = re.match(r'(\d+)\s*Years?\s*and\s*(\d+)\s*Months?', str(value))
        if match:
            return int(match.group(1)) * 12 + int(match.group(2))
        return np.nan

    def _drop_identity(self, df):
        cols_to_drop = [c for c in self.IDENTITY_COLS if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
        return df

    def _clean_numeric_strings(self, df):
        for col in self.NUMERIC_DIRTY:
            if col in df.columns:
                df[col] = df[col].apply(self._strip_underscore)
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    def _replace_placeholders(self, df):
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].replace(self.PLACEHOLDERS, np.nan)
        return df

    def _parse_dates(self, df):
        if 'Credit_History_Age' in df.columns:
            df['Credit_History_Months'] = df['Credit_History_Age'].apply(self._parse_credit_age)
            df = df.drop(columns=['Credit_History_Age'])
        if 'Month' in df.columns and df['Month'].dtype == 'object':
            df['Month'] = df['Month'].map(self.MONTH_MAP)
        return df

    def _handle_outliers(self, df):
        for col, (lo, hi) in self.OUTLIER_RULES.items():
            if col in df.columns:
                if lo is not None:
                    df.loc[df[col] < lo, col] = np.nan
                if hi is not None:
                    df.loc[df[col] > hi, col] = np.nan
        return df

    def _impute(self, df, fit):
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if 'Month' in numeric_cols:
            numeric_cols.remove('Month')

        if fit:
            self.numeric_medians = {c: df[c].median() for c in numeric_cols}
            if 'Month' in df.columns:
                self.month_mode = df['Month'].mode().iloc[0]

        for col in numeric_cols:
            if col in self.numeric_medians:
                df[col] = df[col].fillna(self.numeric_medians[col])

        if 'Month' in df.columns and self.month_mode is not None:
            df['Month'] = df['Month'].fillna(self.month_mode)

        cat_cols = df.select_dtypes(include='object').columns.tolist()
        if self.TARGET_COL in cat_cols:
            cat_cols.remove(self.TARGET_COL)

        if fit:
            self.categorical_modes = {
                c: df[c].mode().iloc[0] if not df[c].mode().empty else 'Unknown'
                for c in cat_cols
            }

        for col in cat_cols:
            if col in self.categorical_modes:
                df[col] = df[col].fillna(self.categorical_modes[col])

        return df

    def _feature_engineer(self, df, fit):
        if 'Type_of_Loan' in df.columns:
            df['Type_of_Loan'] = df['Type_of_Loan'].fillna('None')
            df['Num_Loan_Types'] = df['Type_of_Loan'].apply(
                lambda s: 0 if s == 'None' else len([x for x in re.split(r',|and', s) if x.strip()])
            )
            for loan_type in self.LOAN_TYPES:
                col_name = 'Loan_' + loan_type.replace(' ', '_').replace('-', '_')
                df[col_name] = df['Type_of_Loan'].str.contains(loan_type, regex=False).astype(int)
            df = df.drop(columns=['Type_of_Loan'])

        if 'Credit_Mix' in df.columns:
            df['Credit_Mix'] = df['Credit_Mix'].map(self.CREDIT_MIX_MAP)
            if fit:
                self.credit_mix_median = df['Credit_Mix'].median()
            df['Credit_Mix'] = df['Credit_Mix'].fillna(self.credit_mix_median)

        if 'Payment_of_Min_Amount' in df.columns:
            df['Payment_of_Min_Amount'] = df['Payment_of_Min_Amount'].map({'Yes': 1, 'No': 0})
            if fit:
                mode_series = df['Payment_of_Min_Amount'].mode()
                self.payment_min_mode = mode_series.iloc[0] if not mode_series.empty else 0
            df['Payment_of_Min_Amount'] = df['Payment_of_Min_Amount'].fillna(self.payment_min_mode)

        if 'Outstanding_Debt' in df.columns and 'Annual_Income' in df.columns:
            df['Debt_to_Income'] = df['Outstanding_Debt'] / (df['Annual_Income'] + 1)
        if 'Total_EMI_per_month' in df.columns and 'Monthly_Inhand_Salary' in df.columns:
            df['EMI_to_Salary'] = df['Total_EMI_per_month'] / (df['Monthly_Inhand_Salary'] + 1)

        return df

    def clean_and_engineer(self, df, fit):
        df = df.copy()
        df = self._drop_identity(df)
        df = self._clean_numeric_strings(df)
        df = self._replace_placeholders(df)
        df = self._parse_dates(df)
        df = self._handle_outliers(df)
        df = self._impute(df, fit=fit)
        df = self._feature_engineer(df, fit=fit)
        return df

    def fit_transform(self, df):
        df_clean = self.clean_and_engineer(df, fit=True)
        y = df_clean[self.TARGET_COL] if self.TARGET_COL in df_clean.columns else None
        X = df_clean.drop(columns=[self.TARGET_COL]) if y is not None else df_clean

        self.categorical_features = X.select_dtypes(include='object').columns.tolist()
        self.numeric_features = X.select_dtypes(include=np.number).columns.tolist()
        self.feature_columns = X.columns.tolist()

        self.column_transformer = ColumnTransformer(transformers=[
            ('num', StandardScaler(), self.numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical_features),
        ])

        X_transformed = self.column_transformer.fit_transform(X)
        return X_transformed, y

    def transform(self, df):
        df_clean = self.clean_and_engineer(df, fit=False)
        y = df_clean[self.TARGET_COL] if self.TARGET_COL in df_clean.columns else None
        X = df_clean.drop(columns=[self.TARGET_COL]) if y is not None else df_clean

        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_columns]

        X_transformed = self.column_transformer.transform(X)
        return X_transformed, y

    def save(self, path, compress=3):
        joblib.dump(self, path, compress=compress)

    @classmethod
    def load(cls, path):
        return joblib.load(path)

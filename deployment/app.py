import sys
import os
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(APP_DIR))

from inference import CreditScorePredictor, get_sample_test_cases


st.set_page_config(
    page_title='Credit Score Predictor',
    layout='wide',
)


@st.cache_resource
def load_predictor():
    return CreditScorePredictor(models_dir=PROJECT_ROOT / 'models')


predictor = load_predictor()
test_cases = get_sample_test_cases()


OCCUPATIONS = [
    'Scientist', 'Teacher', 'Engineer', 'Entrepreneur', 'Developer',
    'Lawyer', 'Media_Manager', 'Doctor', 'Journalist', 'Manager',
    'Accountant', 'Musician', 'Mechanic', 'Writer', 'Architect',
]
PAYMENT_BEHAVIOURS = [
    'High_spent_Small_value_payments', 'High_spent_Medium_value_payments',
    'High_spent_Large_value_payments', 'Low_spent_Small_value_payments',
    'Low_spent_Medium_value_payments', 'Low_spent_Large_value_payments',
]
LOAN_TYPES = [
    'Personal Loan', 'Student Loan', 'Mortgage Loan', 'Auto Loan',
    'Payday Loan', 'Credit-Builder Loan', 'Home Equity Loan',
    'Debt Consolidation Loan', 'Not Specified',
]
MONTHS = [
    'January', 'February', 'March', 'April',
    'May', 'June', 'July', 'August',
]


def init_state():
    defaults = test_cases['Standard']
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if 'last_prediction' not in st.session_state:
        st.session_state['last_prediction'] = None


def apply_preset(preset_name):
    preset = test_cases[preset_name]
    for key, value in preset.items():
        st.session_state[key] = value
    st.session_state['last_prediction'] = None


def format_type_of_loan(loan_list):
    if not loan_list:
        return 'Not Specified'
    if len(loan_list) == 1:
        return loan_list[0]
    return ', '.join(loan_list[:-1]) + ', and ' + loan_list[-1]


def parse_type_of_loan(loan_string):
    if not loan_string or loan_string.strip() == 'Not Specified':
        return ['Not Specified']
    cleaned = loan_string.replace(', and ', ', ').replace(' and ', ', ')
    parts = [p.strip() for p in cleaned.split(',') if p.strip()]
    return [p for p in parts if p in LOAN_TYPES] or ['Not Specified']


init_state()


with st.sidebar:
    st.title('Credit Score Predictor')
    st.caption('Model: RandomForest | F1-macro: 0.7245')

    st.divider()
    st.subheader('Sample Test Cases')
    st.caption('Klik buat auto-fill form dengan profil representatif tiap kelas.')

    col_a, col_b, col_c = st.columns(3)
    if col_a.button('Good', use_container_width=True):
        apply_preset('Good')
        st.rerun()
    if col_b.button('Standard', use_container_width=True):
        apply_preset('Standard')
        st.rerun()
    if col_c.button('Poor', use_container_width=True):
        apply_preset('Poor')
        st.rerun()

    st.divider()
    st.subheader('About')
    st.caption(
        'Aplikasi ini memprediksi credit score nasabah '
        '(Good / Standard / Poor) berdasarkan profil keuangan. '
        'Model dilatih pada dataset C (25.000 nasabah).'
    )


st.title('Credit Score Prediction')
st.write(
    'Isi profil nasabah di bawah, atau pilih sample test case di sidebar, '
    'lalu klik **Predict Credit Score**.'
)

with st.form('credit_form', clear_on_submit=False):
    tab1, tab2, tab3, tab4 = st.tabs([
        'Personal & Income',
        'Accounts & Loans',
        'Payment Behavior',
        'Credit History',
    ])

    with tab1:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.selectbox('Month', MONTHS, key='Month')
            st.number_input('Age', min_value=14, max_value=100, step=1, key='Age')
        with c2:
            st.selectbox('Occupation', OCCUPATIONS, key='Occupation')
            st.number_input('Annual Income ($)', min_value=0.0, max_value=500000.0, step=1000.0, key='Annual_Income')
        with c3:
            st.number_input('Monthly Inhand Salary ($)', min_value=0.0, max_value=30000.0, step=100.0, key='Monthly_Inhand_Salary')
            st.number_input('Monthly Balance ($)', min_value=0.0, max_value=5000.0, step=10.0, key='Monthly_Balance')

    with tab2:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.number_input('Num Bank Accounts', min_value=0, max_value=20, step=1, key='Num_Bank_Accounts')
            st.number_input('Num Credit Cards', min_value=0, max_value=20, step=1, key='Num_Credit_Card')
        with c2:
            st.number_input('Interest Rate (%)', min_value=0, max_value=50, step=1, key='Interest_Rate')
            st.number_input('Num of Loan', min_value=0, max_value=15, step=1, key='Num_of_Loan')
        with c3:
            st.number_input('Outstanding Debt ($)', min_value=0.0, max_value=5000.0, step=10.0, key='Outstanding_Debt')
            st.number_input('Total EMI / month ($)', min_value=0.0, max_value=2000.0, step=10.0, key='Total_EMI_per_month')

        current_loans = parse_type_of_loan(st.session_state.get('Type_of_Loan', 'Not Specified'))
        selected_loans = st.multiselect(
            'Type of Loan (pilih semua yang dimiliki)',
            options=LOAN_TYPES,
            default=current_loans,
        )
        st.session_state['Type_of_Loan'] = format_type_of_loan(selected_loans)

    with tab3:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.number_input('Delay from Due Date (days)', min_value=-10, max_value=70, step=1, key='Delay_from_due_date')
            st.number_input('Num of Delayed Payment', min_value=0, max_value=50, step=1, key='Num_of_Delayed_Payment')
        with c2:
            st.selectbox('Payment of Min Amount', ['Yes', 'No'], key='Payment_of_Min_Amount')
            st.selectbox('Payment Behaviour', PAYMENT_BEHAVIOURS, key='Payment_Behaviour')
        with c3:
            st.number_input('Amount Invested Monthly ($)', min_value=0.0, max_value=10000.0, step=10.0, key='Amount_invested_monthly')
            st.number_input('Changed Credit Limit (%)', min_value=-10.0, max_value=40.0, step=0.5, key='Changed_Credit_Limit')

    with tab4:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.selectbox('Credit Mix', ['Bad', 'Standard', 'Good'], key='Credit_Mix')
            st.number_input('Num Credit Inquiries', min_value=0.0, max_value=50.0, step=1.0, key='Num_Credit_Inquiries')
        with c2:
            st.number_input('Credit Utilization Ratio (%)', min_value=0.0, max_value=50.0, step=0.5, key='Credit_Utilization_Ratio')
        with c3:
            current_ch = st.session_state.get('Credit_History_Age', '18 Years and 0 Months')
            try:
                parts = current_ch.split(' Years and ')
                years_default = int(parts[0])
                months_default = int(parts[1].split(' Months')[0])
            except Exception:
                years_default, months_default = 18, 0
            years = st.number_input('Credit History (Years)', min_value=0, max_value=35, value=years_default, step=1)
            months = st.number_input('Credit History (Months)', min_value=0, max_value=11, value=months_default, step=1)
            st.session_state['Credit_History_Age'] = f'{years} Years and {months} Months'

    submitted = st.form_submit_button('Predict Credit Score', use_container_width=True, type='primary')


if submitted:
    input_dict = {
        field: st.session_state[field]
        for field in CreditScorePredictor.REQUIRED_FIELDS
    }
    try:
        result = predictor.predict(input_dict)
        st.session_state['last_prediction'] = result
        st.session_state['last_input'] = input_dict
    except Exception as exc:
        st.error(f'Prediction error: {exc}')
        st.session_state['last_prediction'] = None


result = st.session_state.get('last_prediction')
if result:
    st.divider()
    st.subheader('Prediction Result')

    pred_class = result['predicted_class']
    confidence = result['confidence']
    probabilities = result['probabilities']

    c1, c2 = st.columns([1, 2])

    with c1:
        st.metric(
            label='Predicted Credit Score',
            value=pred_class,
            delta=f'{confidence*100:.1f}% confidence',
        )

    with c2:
        proba_df = pd.DataFrame({
            'Class': list(probabilities.keys()),
            'Probability': list(probabilities.values()),
        }).sort_values('Probability', ascending=False)
        st.bar_chart(proba_df.set_index('Class')['Probability'])

    with st.expander('Probability detail (per class)'):
        for cls in ['Good', 'Standard', 'Poor']:
            prob = probabilities.get(cls, 0.0)
            st.progress(prob, text=f'{cls}: {prob*100:.2f}%')

    with st.expander('Input recap'):
        st.json(st.session_state.get('last_input', {}))

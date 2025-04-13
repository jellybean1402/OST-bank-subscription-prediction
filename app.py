import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, jaccard_score, roc_auc_score, precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
import joblib
import streamlit as st


st.title("Bank Subscription Prediction")
# Load the deployed model
MODEL_PATH = "./models/deployed_model.pkl"
model = joblib.load(MODEL_PATH)

def main():
    age = st.slider("Age:",min_value = 18, max_value = 95, value = 50, step = 1)
    balance = st.slider("Balance:",min_value = -8019, max_value = 81204, value = 0, step = 1)
    day = st.slider("Day:",min_value = 1, max_value = 31, value = 5, step = 1)
    duration = st.slider("Duration:",min_value = 0, max_value = 4918, value = 20, step = 1)
    campaign = st.slider("Campaign:",min_value = 1, max_value = 63, value = 1, step = 1)
    pdays = st.slider("Pdays:",min_value = -1, max_value = 871, value = 5, step = 1)
    previous = st.slider("Previous:",min_value = 0, max_value = 37, value = 1, step = 1)
    job = st.selectbox("Select Job:", ["management", "technician", "entrepreneur", "blue_collar", "unknown", "retired", "admin", "services", "self_employed", "unemployed", "housemaid", "student"])
    marital_status = st.selectbox("Select Marital Status:", ["married", "single", "divorced"])
    education = st.selectbox("Select Education:", ["primary", "secondary", "tertiary", "unknown"])
    contact = st.selectbox("Mode of contact", ["unknown", "cellular", "telephone"])
    month = st.selectbox("Month", ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    poutcome = st.selectbox("Poutcome", ["unknown", "success", "failure", "other"])
    default = st.checkbox("Default:")
    housing = st.checkbox("Housing Loan:")
    loan = st.checkbox("Personal Loan:")


    age = age
    balance = balance
    day = day
    duration = duration
    campaign = campaign
    pdays = pdays
    previous = previous

    if job == "admin":
        job_admin = True
        job_blue_collar = False
        job_entrepreneur = False
        job_housemaid = False
        job_management = False
        job_retired = False
        job_self_employed = False
        job_services = False
        job_student = False
        job_technician = False
        job_unemployed = False
        job_unknown = False
    elif job == "blue_collar":
        job_admin = False
        job_blue_collar = True
        job_entrepreneur = False
        job_housemaid = False
        job_management = False
        job_retired = False
        job_self_employed = False
        job_services = False
        job_student = False
        job_technician = False
        job_unemployed = False
        job_unknown = False
    elif job == "entrepreneur":
        job_admin = False
        job_blue_collar = False
        job_entrepreneur = True
        job_housemaid = False
        job_management = False
        job_retired = False
        job_self_employed = False
        job_services = False
        job_student = False
        job_technician = False
        job_unemployed = False
        job_unknown = False
    elif job == "housemaid":
        job_admin = False
        job_blue_collar = False
        job_entrepreneur = False
        job_housemaid = True
        job_management = False
        job_retired = False
        job_self_employed = False
        job_services = False
        job_student = False
        job_technician = False
        job_unemployed = False
        job_unknown = False
    elif job == "management":
        job_admin = False
        job_blue_collar = False
        job_entrepreneur = False
        job_housemaid = False
        job_management = True
        job_retired = False
        job_self_employed = False
        job_services = False
        job_student = False
        job_technician = False
        job_unemployed = False
        job_unknown = False
    elif job == "retired":
        job_admin = False
        job_blue_collar = False
        job_entrepreneur = False
        job_housemaid = False
        job_management = False
        job_retired = True
        job_self_employed = False
        job_services = False
        job_student = False
        job_technician = False
        job_unemployed = False
        job_unknown = False
    elif job == "self_employed":
        job_admin = False
        job_blue_collar = False
        job_entrepreneur = False
        job_housemaid = False
        job_management = False
        job_retired = False
        job_self_employed = True
        job_services = False
        job_student = False
        job_technician = False
        job_unemployed = False
        job_unknown = False
    elif job == "services":
        job_admin = False
        job_blue_collar = False
        job_entrepreneur = False
        job_housemaid = False
        job_management = False
        job_retired = False
        job_self_employed = False
        job_services = True
        job_student = False
        job_technician = False
        job_unemployed = False
        job_unknown = False
    elif job == "student":
        job_admin = False
        job_blue_collar = False
        job_entrepreneur = False
        job_housemaid = False
        job_management = False
        job_retired = False
        job_self_employed = False
        job_services = False
        job_student = True
        job_technician = False
        job_unemployed = False
        job_unknown = False
    elif job == "technician":
        job_admin = False
        job_blue_collar = False
        job_entrepreneur = False
        job_housemaid = False
        job_management = False
        job_retired = False
        job_self_employed = False
        job_services = False
        job_student = False
        job_technician = True
        job_unemployed = False
        job_unknown = False
    elif job == "unemployed":
        job_admin = False
        job_blue_collar = False
        job_entrepreneur = False
        job_housemaid = False
        job_management = False
        job_retired = False
        job_self_employed = False
        job_services = False
        job_student = False
        job_technician = False
        job_unemployed = True
        job_unknown = False
    elif job == "unknown":
        job_admin = False
        job_blue_collar = False
        job_entrepreneur = False
        job_housemaid = False
        job_management = False
        job_retired = False
        job_self_employed = False
        job_services = False
        job_student = False
        job_technician = False
        job_unemployed = False
        job_unknown = True

    if marital_status == "divorced":
        marital_divorced = True
        marital_married = False
        marital_single = False
    elif marital_status == "married":
        marital_divorced = False
        marital_married = True
        marital_single = False
    elif marital_status == "single":
        marital_divorced = False
        marital_married = False
        marital_single = True

    if education == "unknown":
        education_0 = True
        education_1 = False
        education_2 = False
        education_3 = False
    elif education == "primary":
        education_0 = False
        education_1 = True
        education_2 = False
        education_3 = False
    elif education == "secondary":
        education_0 = False
        education_1 = False
        education_2 = True
        education_3 = False
    elif education == "tertiary":
        education_0 = False
        education_1 = False
        education_2 = False
        education_3 = True

    if default == True:
        default_yes = True
        default_no = False
    elif default == False:
        default_yes = False
        default_no = True

    if housing == True:
        housing_yes = True
        housing_no = False
    elif housing == False:
        housing_yes = False
        housing_no = True

    if loan == True:
        loan_yes = True
        loan_no = False
    elif loan == False:
        loan_yes = False
        loan_no = True

    if contact == "cellular":
        contact_cellular = True
        contact_telephone = False
        contact_unknown = False
    elif contact == "telephone":
        contact_cellular = False
        contact_telephone = True
        contact_unknown = False
    elif contact == "unknown":
        contact_cellular = False
        contact_telephone = False
        contact_unknown = True

    if month == "Jan":
        month_jan = True
        month_feb = False
        month_mar = False
        month_apr = False
        month_may = False
        month_jun = False
        month_jul = False
        month_aug = False
        month_sep = False
        month_oct = False
        month_nov = False
        month_dec = False
    elif month == "Feb":
        month_jan = False
        month_feb = True
        month_mar = False
        month_apr = False
        month_may = False
        month_jun = False
        month_jul = False
        month_aug = False
        month_sep = False
        month_oct = False
        month_nov = False
        month_dec = False
    elif month == "Mar":
        month_jan = False
        month_feb = False
        month_mar = True
        month_apr = False
        month_may = False
        month_jun = False
        month_jul = False
        month_aug = False
        month_sep = False
        month_oct = False
        month_nov = False
        month_dec = False
    elif month == "Apr":
        month_jan = False
        month_feb = False
        month_mar = False
        month_apr = True
        month_may = False
        month_jun = False
        month_jul = False
        month_aug = False
        month_sep = False
        month_oct = False
        month_nov = False
        month_dec = False
    elif month == "May":
        month_jan = False
        month_feb = False
        month_mar = False
        month_apr = False
        month_may = True
        month_jun = False
        month_jul = False
        month_aug = False
        month_sep = False
        month_oct = False
        month_nov = False
        month_dec = False
    elif month == "Jun":
        month_jan = False
        month_feb = False
        month_mar = False
        month_apr = False
        month_may = False
        month_jun = True
        month_jul = False
        month_aug = False
        month_sep = False
        month_oct = False
        month_nov = False
        month_dec = False
    elif month == "Jul":
        month_jan = False
        month_feb = False
        month_mar = False
        month_apr = False
        month_may = False
        month_jun = False
        month_jul = True
        month_aug = False
        month_sep = False
        month_oct = False
        month_nov = False
        month_dec = False
    elif month == "Aug":
        month_jan = False
        month_feb = False
        month_mar = False
        month_apr = False
        month_may = False
        month_jun = False
        month_jul = False
        month_aug = True
        month_sep = False
        month_oct = False
        month_nov = False
        month_dec = False
    elif month == "Sep":
        month_jan = False
        month_feb = False
        month_mar = False
        month_apr = False
        month_may = False
        month_jun = False
        month_jul = False
        month_aug = False
        month_sep = True
        month_oct = False
        month_nov = False
        month_dec = False
    elif month == "Oct":
        month_jan = False
        month_feb = False
        month_mar = False
        month_apr = False
        month_may = False
        month_jun = False
        month_jul = False
        month_aug = False
        month_sep = False
        month_oct = True
        month_nov = False
        month_dec = False
    elif month == "Nov":
        month_jan = False
        month_feb = False
        month_mar = False
        month_apr = False
        month_may = False
        month_jun = False
        month_jul = False
        month_aug = False
        month_sep = False
        month_oct = True
        month_nov = False
        month_dec = False
    elif month == "Nov":
        month_jan = False
        month_feb = False
        month_mar = False
        month_apr = False
        month_may = False
        month_jun = False
        month_jul = False
        month_aug = False
        month_sep = False
        month_oct = False
        month_nov = True
        month_dec = False
    elif month == "Dec":
        month_jan = False
        month_feb = False
        month_mar = False
        month_apr = False
        month_may = False
        month_jun = False
        month_jul = False
        month_aug = False
        month_sep = False
        month_oct = False
        month_nov = False
        month_dec = True

    if poutcome == "failure":
        poutcome_failure = True
        poutcome_other = False
        poutcome_success = False
        poutcome_unknown = False
    elif poutcome == "other":
        poutcome_failure = False
        poutcome_other = True
        poutcome_success = False
        poutcome_unknown = False
    elif poutcome == "success":
        poutcome_failure = False
        poutcome_other = False
        poutcome_success = True
        poutcome_unknown = False
    elif poutcome == "unknown":
        poutcome_failure = False
        poutcome_other = False
        poutcome_success = False
        poutcome_unknown = True

    dict = {
        "age":[age],
        "balance":[balance],
        "day":[day],
        "duration":[duration],
        "campaign":[campaign],
        "pdays":[pdays],
        "previous":[previous],
        "job_admin":[job_admin],
        "job_blue_collar":[job_blue_collar],
        "job_entrepreneur":[job_entrepreneur],
        "job_housemaid":[job_housemaid],
        "job_management":[job_management],
        "job_retired":[job_retired],
        "job_self_employed":[job_self_employed],
        "job_services":[job_services],
        "job_student":[job_student],
        "job_technician":[job_technician],
        "job_unemployed":[job_unemployed],
        "job_unknown":[job_unknown],
        "marital_divorced":[marital_divorced],
        "marital_married":[marital_married],
        "marital_single":[marital_single],
        "education_0":[education_0],
        "education_1": [education_1],
        "education_2": [education_2],
        "education_3": [education_3],
        "default_no":[default_no],
        "default_yes":[default_yes],
        "housing_no":[housing_no],
        "housing_yes":[housing_yes],
        "loan_no":[loan_no],
        "loan_yes":[loan_yes],
        "contact_cellular":[contact_cellular],
        "contact_telephone":[contact_telephone],
        "contact_unknown":[contact_unknown],
        "month_apr":[month_apr],
        "month_aug":[month_aug],
        "month_dec":[month_dec],
        "month_feb":[month_feb],
        "month_jan":[month_jan],
        "month_jul":[month_jul],
        "month_jun":[month_jun],
        "month_mar":[month_mar],
        "month_may":[month_may],
        "month_nov":[month_nov],
        "month_oct":[month_oct],
        "month_sep":[month_sep],
        "poutcome_failure":[poutcome_failure],
        "poutcome_other":[poutcome_other],
        "poutcome_success":[poutcome_success],
        "poutcome_unknown":[poutcome_unknown]
    }

    df = pd.DataFrame(dict)
    prediction = model.predict(df)

    st.header("Prediction:")
    if prediction == 0:
        st.error("Client will not subscribe")
    elif prediction == 1:
        st.success("Client will subscribe")


if __name__ == '__main__':
    main()
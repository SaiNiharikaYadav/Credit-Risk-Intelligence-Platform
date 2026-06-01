import streamlit as st
import pandas as pd
import joblib
from PIL import Image
import os
from dotenv import load_dotenv
import google.generativeai as genai
import joblib


model = joblib.load("models/xgboost_model.pkl")

original_features = joblib.load(
    "models/original_features.pkl"
)

model_columns = joblib.load(
    "models/model_columns.pkl"
)

numeric_defaults = joblib.load(
    "models/numeric_defaults.pkl"
)

categorical_defaults = joblib.load(
    "models/categorical_defaults.pkl"
)
load_dotenv()

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

gemini_model = genai.GenerativeModel("models/gemini-2.5-flash")

st.set_page_config(
    page_title="Credit Risk Intelligence Platform",
    layout="wide"
)

# Sidebar
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "EDA Dashboard",
        "Risk Prediction",
        "Explainability",
        "Business Rules",
        "Chatbot"
    ]
)

# HOME
if page == "Home":

    st.title("🏦 Credit Risk Intelligence Platform")

    st.success("XGBoost + SHAP Credit Risk System")

    st.markdown("""
    ### Features

    - Exploratory Data Analysis
    - Credit Risk Prediction
    - Explainable AI (SHAP)
    - Business Rules Engine
    - Talk-to-Data Chatbot
    """)

    st.subheader("Model Information")

    st.write("Model : XGBoost")
    st.write("ROC-AUC : 0.7527")

# EDA PAGE
elif page == "EDA Dashboard":

    st.title("📊 EDA Dashboard")

    df = pd.read_csv("data/application_train.csv")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    with col3:
        st.metric("Default Rate", f"{df['TARGET'].mean()*100:.2f}%")

    st.subheader("Target Distribution")
    st.bar_chart(df["TARGET"].value_counts())

    st.subheader("Missing Values (Top 10)")
    missing = df.isnull().sum().sort_values(ascending=False).head(10)
    st.bar_chart(missing)

# PREDICTION PAGE
elif page == "Risk Prediction":

    st.title("🔮 Credit Risk Prediction")

    st.write(
        "Predict customer default probability using the trained XGBoost model."
    )

    # =====================================
    # Applicant Profile
    # =====================================

    st.subheader("Applicant Profile")

    education = st.selectbox(
        "Education Type",
        [
            "Higher education",
            "Secondary / secondary special",
            "Incomplete higher",
            "Lower secondary"
        ]
    )

    gender = st.selectbox(
        "Gender",
        [
            "Male",
            "Female"
        ]
    )

    income_type = st.selectbox(
        "Income Type",
        [
            "Working",
            "Commercial associate",
            "State servant",
            "Pensioner"
        ]
    )

    # =====================================
    # Top Risk Features
    # =====================================

    st.subheader("Top Risk Features")

    top_features = [
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_GOODS_PRICE",
        "AMT_ANNUITY",
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "CNT_CHILDREN",
        "CNT_FAM_MEMBERS",
        "REGION_RATING_CLIENT",
        "REGION_RATING_CLIENT_W_CITY",
        "OWN_CAR_AGE",
        "OBS_30_CNT_SOCIAL_CIRCLE",
        "DEF_30_CNT_SOCIAL_CIRCLE",
        "OBS_60_CNT_SOCIAL_CIRCLE",
        "DEF_60_CNT_SOCIAL_CIRCLE",
        "AMT_REQ_CREDIT_BUREAU_YEAR",
        "AMT_REQ_CREDIT_BUREAU_MON"
    ]

    user_data = {}

    for feature in top_features:

        default_value = 0.0

        if feature in numeric_defaults:
            default_value = float(numeric_defaults[feature])

        user_data[feature] = st.number_input(
            feature,
            value=default_value
        )

    # =====================================
    # Predict Button
    # =====================================

    if st.button("Predict Risk"):

        full_input = {}

        for feature in original_features:

            if feature in user_data:

                full_input[feature] = user_data[feature]

            elif feature in numeric_defaults:

                full_input[feature] = numeric_defaults[feature]

            elif feature in categorical_defaults:

                full_input[feature] = categorical_defaults[feature]

            else:

                full_input[feature] = 0

        # =====================================
        # User-selected categorical values
        # =====================================

        full_input["NAME_EDUCATION_TYPE"] = education

        full_input["CODE_GENDER"] = (
            "M" if gender == "Male" else "F"
        )

        full_input["NAME_INCOME_TYPE"] = income_type

        # =====================================
        # DataFrame
        # =====================================

        input_df = pd.DataFrame([full_input])

        input_df = pd.get_dummies(input_df)

        input_df = input_df.reindex(
            columns=model_columns,
            fill_value=0
        )

        # =====================================
        # Prediction
        # =====================================

        probability = model.predict_proba(
            input_df
        )[0][1]

        risk_score = round(
            probability * 100,
            2
        )

        if risk_score < 30:

            risk_band = "🟢 LOW RISK"

        elif risk_score < 70:

            risk_band = "🟡 MEDIUM RISK"

        else:

            risk_band = "🔴 HIGH RISK"

        st.metric(
            "Default Probability",
            f"{risk_score:.2f}%"
        )

        st.metric(
            "Risk Score",
            f"{risk_score:.2f}"
        )

        st.success(
            f"Risk Band: {risk_band}"
        )


# EXPLAINABILITY
elif page == "Explainability":

    st.title("🔍 SHAP Explainability")

    st.write("Top Features Influencing Credit Default Risk")

    try:
        image = Image.open("documents/shap_summary.png")
        st.image(image, use_container_width=True)

        st.subheader("📌 Key Insights")

        st.markdown("""
        ### Understanding the SHAP Plot

        SHAP (SHapley Additive exPlanations) helps explain how each feature
        contributes to loan default predictions.

        **Key observations:**
        - Higher credit amounts generally increase default risk.
        - Lower income levels are associated with higher risk.
        - Previous credit history significantly impacts predictions.
        - Bureau-related features are strong indicators of creditworthiness.
        - Employment and financial stability features influence repayment behavior.

        ### Business Value
        - Improves transparency of AI decisions.
        - Helps loan officers understand risk factors.
        - Supports regulatory compliance and explainable lending.
        - Builds trust in automated credit decisions.
        """)

    except Exception as e:
        st.warning("Run shap_analysis.py first")
        st.error(str(e))

# BUSINESS RULES
elif page == "Business Rules":

    st.title("📋 Business Rules Engine")

    st.write(
        """
        Business rules provide additional guardrails
        alongside machine learning predictions.
        """
    )

    st.warning(
        """
        Rule 1:
        If Credit Amount > 5 × Annual Income
        → Flag as High Risk
        """
    )

    st.warning(
        """
        Rule 2:
        If EXT_SOURCE_3 < 0.20
        → Increase Risk Level
        """
    )

    st.warning(
        """
        Rule 3:
        If Employment History < 1 Year
        → Medium Risk Review
        """
    )

    st.warning(
        """
        Rule 4:
        If Bureau Enquiries > 5
        → Manual Verification Required
        """
    )
    st.warning(
    """
    Rule 5:
    If AMT_REQ_CREDIT_BUREAU_YEAR > 5
    → Manual Review Required
    """
    )

    st.success(
    """
    Final Lending Decision Pipeline:

    1. XGBoost Risk Prediction
    2. SHAP Explainability
    3. Business Rules Validation
    4. Final Risk Classification
    """
    )

# CHATBOT
elif page == "Chatbot":

    st.title("💬 Talk To Data System")

    question = st.text_input(
        "Ask a business question"
    )

    if st.button("Ask"):

        from src.chatbot.sql_generator import generate_sql
        from src.chatbot.query_runner import run_query

        sql = generate_sql(question)

        st.subheader("Generated SQL")
        st.code(sql, language="sql")

        # Validate generated SQL
        if not sql:
            st.error("No SQL generated.")
            st.stop()

        if sql.startswith("-- ERROR"):
            st.error(sql)
            st.stop()

        result = run_query(sql)

        st.subheader("Results")
        st.dataframe(result)

        # Business Insight
        st.subheader("Business Insight")

        if not result.empty:
            count = result.iloc[0, 0]

            st.success(
                f"There are {count:,} customers marked as default cases."
            )
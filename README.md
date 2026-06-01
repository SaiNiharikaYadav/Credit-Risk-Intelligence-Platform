Credit Risk Intelligence Platform
Overview

Credit Risk Intelligence Platform is an AI-powered web application developed to analyze customer credit risk, predict loan defaults, explain model predictions, and provide business insights through natural language queries.

The platform integrates Machine Learning, Explainable AI (SHAP), Business Rules, SQLite, Streamlit, and Google Gemini to support intelligent credit risk analysis.

Features
Home Dashboard
Overview of the Credit Risk Intelligence Platform
Navigation to all modules
Exploratory Data Analysis (EDA)
Income distribution analysis
Credit amount analysis
Target variable distribution
Interactive visualizations
Risk Prediction
Predict customer default risk
Uses trained XGBoost model
Returns risk probability and prediction
Explainability
SHAP-based feature importance analysis
Model interpretation and transparency
Helps understand prediction behavior
Business Rules Engine
Rule-based customer risk assessment
Identifies high-risk applicants
Supports business decision-making
AI Chatbot
Natural Language to SQL conversion
Powered by Google Gemini
Executes SQL queries on SQLite database
Returns business insights instantly

Example Questions:

How many customers defaulted?
What is the average income of customers?
How many customers have credit greater than 1000000?
What is the maximum credit amount?
How many customers have income greater than 300000?
Dataset

The project uses data from the Home Credit Default Risk dataset.

Datasets included:

application_train.csv
bureau.csv
previous_application.csv
installments_payments.csv

These datasets are used for:

Risk prediction
Business analytics
Credit behavior analysis
SQL-based querying


Technology Stack

| Component        | Technology          |
| ---------------- | ------------------- |
| Frontend         | Streamlit           |
| Machine Learning | XGBoost             |
| Data Processing  | Pandas, NumPy       |
| Visualization    | Matplotlib, Seaborn |
| Explainability   | SHAP                |
| Database         | SQLite              |
| LLM              | Google Gemini       |
| Deployment       | Docker              |
| Version Control  | Git & GitHub        |



Step 1: Data Collection
Load customer credit datasets.

Step 2: Data Preprocessing
Handle missing values and prepare features.

Step 3: Model Training
Train XGBoost model for credit risk prediction.

Step 4: Database Creation
Create SQLite database from datasets.

Step 5: Risk Assessment
Generate customer default predictions.

Step 6: Explainability
Generate SHAP feature importance explanations.

Step 7: Business Rules
Apply predefined business logic.

Step 8: AI Chatbot
Convert natural language questions into SQL queries and retrieve results from the database.

Project Structure
CreditRiskPlatform
│
├── data
│   ├── application_train.csv
│   ├── bureau.csv
│   ├── previous_application.csv
│   └── installments_payments.csv
│
├── models
│   ├── xgboost_model.pkl
│   ├── feature_columns.pkl
│   ├── feature_importance.csv
│   └── other preprocessing files
│
├── screenshots
│
├── sql
│   └── schema.sql
│
├── src
│   ├── chatbot
│   ├── data
│   ├── explainability
│   └── ml
│
├── app.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
Installation
Clone Repository
git clone https://github.com/SaiNiharikaYadav/Credit-Risk-Intelligence-Platform.git

cd Credit-Risk-Intelligence-Platform
Create Virtual Environment
python -m venv venv
Windows
venv\Scripts\activate
Install Dependencies
pip install -r requirements.txt
Configure Environment Variables

Create a .env file:

GEMINI_API_KEY=YOUR_GEMINI_API_KEY
Run Application
streamlit run app.py

Open:

http://localhost:8501
Docker Deployment
Build and Run
docker-compose up --build

Open:

http://localhost:8501
Stop Containers
docker-compose down


Application Screenshots
Home Page
EDA Dashboard
Risk Prediction
Explainability
Business Rules
AI Chatbot

Future Improvements
Cloud Deployment
User Authentication
REST API Integration
Real-Time Monitoring
Model Retraining Pipeline
Advanced Business Intelligence Dashboard


Author
Niharika Yadav

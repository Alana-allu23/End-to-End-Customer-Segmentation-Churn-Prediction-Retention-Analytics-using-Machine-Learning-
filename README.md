End-to-End Customer Churn Prediction & Retention Analytics

An end-to-end machine learning project that identifies telecom customers who are likely to leave, explains the factors driving churn, groups customers into meaningful segments, and provides data-driven retention recommendations.

Project Overview

Customer churn directly affects revenue and customer acquisition costs. This project uses the IBM Telco Customer Churn dataset to build a complete analytics and prediction system that can help a telecom company:

Identify customers at high risk of churn

Understand the main factors influencing churn

Segment customers based on their behavior and value

Recommend targeted retention strategies

Explore results through an interactive Streamlit application

Present business insights through a Power BI dashboard

Business Problem

Acquiring a new customer is usually more expensive than retaining an existing one. The goal of this project is not only to predict whether a customer may churn, but also to convert the prediction into useful business actions.

The main questions addressed are:

Which customers are most likely to churn?

What customer characteristics are associated with churn?

Which customer groups require immediate attention?

What retention action can be offered to each high-risk customer?

Dataset

Source: IBM Telco Customer Churn dataset

Records: 7,043 customers

Original features: 20 input variables and 1 target variable

Target: Churn (Yes or No)

The dataset includes:

Customer demographics

Account and tenure information

Subscribed services

Contract and payment information

Monthly and total charges

Churn status

Project Workflow

flowchart TD
    A[Data Collection] --> B[Data Cleaning and EDA]
    B --> C[Feature Engineering]
    C --> D[Preprocessing and Encoding]
    D --> E[Train-Test Split]
    E --> F[SMOTE on Training Data]
    F --> G[Model Training and Tuning]
    G --> H[Evaluation and Explainability]
    H --> I[Customer Segmentation]
    I --> J[Retention Analytics]
    J --> K[Streamlit App and Power BI]

SMOTE is applied only to the training data. The test data remains untouched to provide a fair evaluation of the model.

Main Project Stages

1. Data Cleaning

Converted TotalCharges to a numeric data type

Handled blank and missing values

Checked duplicate records

Standardized column values and data types

Reviewed numerical and categorical features

2. Exploratory Data Analysis

The analysis investigates churn across:

Contract type

Customer tenure

Internet service

Online security and technical support

Payment method

Monthly charges

Senior citizen status

Paperless billing

3. Data Preprocessing

Binary encoding for Yes/No variables

One-hot encoding for multi-category variables

Feature scaling for numerical variables

Stratified train-test split

SMOTE applied only to the training set

4. Machine Learning Models

The following classification models were compared:

Logistic Regression

Decision Tree

Random Forest

Support Vector Machine

K-Nearest Neighbors

XGBoost

The final model was selected using business-relevant evaluation metrics rather than accuracy alone.

Model Evaluation

The models were evaluated using:

Accuracy: Overall percentage of correct predictions

Precision: Reliability of predicted churn cases

Recall: Ability to identify customers who actually churn

F1-score: Balance between precision and recall

ROC-AUC: Ability to separate churners from non-churners across thresholds

Confusion matrix: Breakdown of correct and incorrect predictions

For churn prediction, recall, F1-score, and ROC-AUC receive greater attention because missing a real churner may mean losing a customer and future revenue.

Final Model Results

Add the final values produced by your notebook before publishing:

Metric

Score

Accuracy

Add score

Precision

Add score

Recall

Add score

F1-score

Add score

ROC-AUC

Add score

Model Explainability

Feature importance and SHAP analysis are used to explain:

Why the model predicted that a customer may churn

Which features increase churn risk

Which features reduce churn risk

How individual predictions can be communicated to business teams

Common churn patterns investigated in this project include short tenure, month-to-month contracts, high monthly charges, and the absence of support-related services.

Customer Segmentation

Customer segmentation is performed using clustering to group customers with similar characteristics. The segments can be described using factors such as:

Tenure

Monthly charges

Total charges

Service usage

Churn risk

Customer value

This helps the business design different strategies for high-risk, loyal, new, and high-value customers.

Retention Analytics

The prediction and segmentation results are converted into practical actions:

Customer condition

Suggested retention action

Month-to-month contract and high risk

Offer a discounted annual contract

New customer with short tenure

Provide onboarding and early-support follow-up

High monthly charges

Recommend a suitable lower-cost bundle

No online security or tech support

Offer a free trial or service bundle

High-value customer at risk

Assign priority support and a personalized offer

Streamlit Web Application

The Streamlit application allows a user to:

Enter customer information

Predict churn or non-churn

View churn probability

See the customer's risk level

Receive a suggested retention action

Run the Application Locally

git clone https://github.com/Alana-allu23/End-to-End-Customer-Churn-Prediction-Retention-Analytics-using-Machine-Learning-.git
cd End-to-End-Customer-Churn-Prediction-Retention-Analytics-using-Machine-Learning-

python -m venv .venv

Activate the environment on Windows:

.venv\Scripts\activate

Install the dependencies and start the application:

pip install -r requirements.txt
streamlit run cutomer_churn/app.py

Repository Structure

.
├── cutomer_churn/
│   ├── dataset/
│   ├── app.py
│   ├── churn_model.pkl
│   ├── churn_preprocessor.pkl
│   ├── customer_retention_analytics.csv
│   └── cutomer_seg&churn_retension.ipynb
├── requirements.txt
├── .gitignore
└── README.md

Update the structure above if your current filenames are different. Renaming cutomer_churn to customer_churn and correcting the notebook spelling will make the repository look more professional.

Technologies Used

Python

Pandas and NumPy

Matplotlib and Seaborn

Scikit-learn

Imbalanced-learn

XGBoost

SHAP

Streamlit

Power BI

Jupyter Notebook / VS Code

Git and GitHub

Installation

pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost shap streamlit joblib

To create or update requirements.txt:

pip freeze > requirements.txt

Key Learning Outcomes

Built a complete classification workflow from raw data to deployment

Handled imbalanced data without introducing test-data leakage

Compared multiple models using suitable classification metrics

Explained predictions using feature importance and SHAP

Connected machine learning results with customer retention decisions

Developed an interactive prediction application

Communicated technical findings as actionable business insights

Future Improvements

Deploy the Streamlit application publicly

Add probability-threshold selection based on retention cost

Track model performance and data drift

Add customer lifetime value estimation

Build an automated retention campaign recommendation system

Retrain the model using recent customer behavior data

Author

Alana K

GitHub: Alana-allu23

LinkedIn: Alana K

License

This project is intended for educational and portfolio purposes.

If you find this project useful, consider giving the repository a star.

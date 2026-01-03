import streamlit as st

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv(" Customer-Churn.csv")

st.markdown("<h1 style='text-align:center;'>Customer Churn APP</h1>",unsafe_allow_html=True)
st.markdown("---")

st.subheader("Summary")
st.write("This project predicts whether a customer is likely to leave a company (churn) using machine learning. The goal is to help businesses identify at-risk customers early and take targeted actions to retain them.I collected and processed customer data (such as tenure, contract type, monthly charges, and service usage), handled missing values, encoded categorical features, and scaled numerical variables. Several models were tested — including Logistic Regression, Random Forest, and XGBoost — and XGBoost provided the best performance.The final model predicts churn with strong accuracy and recall, meaning it successfully identifies most customers who are at high risk of leaving. I visualized churn patterns to show why customers leave, helping stakeholders take data-driven retention decisions such as offering discounts, improving support, or modifying plans. project demonstrates the full ML pipeline: data cleaning, feature engineering, model training, evaluation, and deployment through a Streamlit web app for easy use.")
st.subheader("Tools & Techniques")
st.write("Python, Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib/Seaborn, Streamlit")
st.subheader("Download Dataset")
st.write("Download Report")
st.download_button("Download CSV", df.to_csv().encode(), "Customer-Churn.csv")
st.markdown("<h1 style='text-align:center;'>Basic Visualization </h1>",unsafe_allow_html=True)
st.markdown("---")
st.write("Basic data visualizations such as pie charts, bar plots, and histograms were used to understand customer distribution, churn rate, and key patterns in features like contract type, internet service, monthly charges, and senior citizen status")



df = pd.read_csv("C:/Users/Nadish/OneDrive/Desktop/customer churn/Customer-Churn.csv")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.tight_layout(pad=4)
churn_counts = df["Churn"].value_counts()
axes[0, 0].pie(
    churn_counts,
    labels=churn_counts.index,
    autopct='%1.1f%%',
    startangle=90
)
axes[0, 0].set_title("Churn Distribution")
axes[0, 0].axis('equal')


sns.countplot(data=df, x="Contract", hue="Churn", ax=axes[0, 1])
axes[0, 1].set_title("Churn by Contract Type")
axes[0, 1].set_xlabel("Contract")
axes[0, 1].set_ylabel("Customers")
axes[0, 1].tick_params(axis='x', rotation=15)

# 3) Bar — Internet Service vs Churn
sns.countplot(data=df, x="InternetService", hue="Churn", ax=axes[0, 2])
axes[0, 2].set_title("Churn by Internet Service")
axes[0, 2].set_xlabel("Internet Service")
axes[0, 2].set_ylabel("Customers")
axes[0, 2].tick_params(axis='x', rotation=10)


sns.histplot(
    data=df,
    x="MonthlyCharges",
    hue="Churn",
    bins=30,
    kde=True,
    ax=axes[1, 0]
)
axes[1, 0].set_title("Monthly Charges Distribution")
axes[1, 0].set_xlabel("Monthly Charges")


sns.countplot(data=df, x="SeniorCitizen", hue="Churn", ax=axes[1, 1])
axes[1, 1].set_title("Churn by Senior Citizen")
axes[1, 1].set_xlabel("Senior Citizen (0 = No, 1 = Yes)")

sns.countplot(data=df, x="PaymentMethod", hue="Churn", ax=axes[1, 2])
axes[1, 2].set_title("Churn by Payment Method")
axes[1, 2].tick_params(axis='x', rotation=20)

st.pyplot(fig)

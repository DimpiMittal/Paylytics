import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("processed_salary_data.csv")  # Ensure the file path is correct
    return df

df = load_data()

# Display available columns for debugging
st.write("Dataset Columns:", df.columns.tolist())

# Encode categorical columns safely
encoders = {}
categorical_columns = ['experience_level', 'employment_type', 'company_size']

for col in categorical_columns:
    encoders[col] = LabelEncoder()
    df[col] = encoders[col].fit_transform(df[col])


# 🚀 Web App Title & Description
st.title("💰 Welcome to Paylytics")
st.subheader("📊 Analyze salary trends and predict your salary based on experience level, employment type, and company size.")

# Sidebar Navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to:", ["Salary Insights", "Salary Prediction"])

# 📊 **Salary Insights Page**
if page == "Salary Insights":
    st.subheader("📊 Salary Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['salary_in_usd'], bins=30, kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("📈 Salary Trends Over Time")
    df_grouped = df.groupby("work_year")["salary_in_usd"].mean()
    st.line_chart(df_grouped)

    st.subheader("👨‍💻 Salary vs Experience Level")
    fig, ax = plt.subplots()
    sns.boxplot(x=df["experience_level"], y=df["salary_in_usd"], ax=ax)
    st.pyplot(fig)

    st.subheader("🏢 Salary by Company Location")
    location_salary = df.filter(like="company_location_").sum().sort_values(ascending=False)
    st.bar_chart(location_salary)

# 🎯 **Salary Prediction Page**
else:
    st.subheader("💡 Predict Your Salary")

    experience = st.selectbox("Experience Level", df["experience_level"].unique())
    employment_type = st.selectbox("Employment Type", df["employment_type"].unique())
    company_size = st.selectbox("Company Size", df["company_size"].unique())

    # Safe encoding function to handle unseen labels
    def safe_transform(encoder, value):
        if value in encoder.classes_:
            return encoder.transform([value])[0]
        else:
            return -1  # Assign an out-of-range value

    input_data = [[safe_transform(encoders['experience_level'], experience),
                   safe_transform(encoders['employment_type'], employment_type),
                   safe_transform(encoders['company_size'], company_size)]]

    # Train Model
    X = df[categorical_columns]
    y = df['salary_in_usd']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Ensure input_data is accessible inside the button click
    if st.button("Predict Salary"):
        input_df = pd.DataFrame(input_data, columns=categorical_columns)  # Convert input data into DataFrame
        predicted_salary = model.predict(input_df)[0]
        st.success(f"💰 Estimated Salary: ${predicted_salary:,.2f} per year")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("ds_salaries.csv")

# Display first few rows
print(df.head())

print(df.columns)

# Show dataset info
print(df.info())



# Check for missing values
print(df.isnull().sum()) 

# Fill missing values ONLY for numeric columns
df.fillna(df.select_dtypes(include=['number']).mean(), inplace=True)

# Show statistical summary
print(df.describe())

# One-hot encoding for categorical columns
df = pd.get_dummies(df, columns=['job_title', 'company_location'], drop_first=True)

# Show first few rows after encoding
print(df.head())

# Salary Distribution Plot
plt.figure(figsize=(10, 5))  
sns.histplot(df['salary'], bins=30, kde=True, color="blue")
plt.title("Salary Distribution")
plt.xlabel("Salary")
plt.ylabel("Count")
plt.show()

# Salary vs Experience Level
plt.figure(figsize=(8, 5))
sns.boxplot(x="experience_level", y="salary", data=df)
plt.title("Salary vs Experience Level")
plt.show()

# Save the cleaned dataset
df.to_csv("processed_salary_data.csv", index=False)

# This will generate a boxplot which will help to know which roles have highest and lowest salaries
plt.figure(figsize=(12, 6))
sns.boxplot(x="salary_in_usd", y="job_title", data=df)
plt.title("Salary Distribution by Job Title")
plt.xlabel("Salary (USD)")
plt.ylabel("Job Title")
plt.show()

# This will help to know how salaries have changed over time
plt.figure(figsize=(8, 5))
sns.lineplot(x="work_year", y="salary_in_usd", data=df, marker="o", ci=None)
plt.title("Salary Trends Over the Years")
plt.xlabel("Year")
plt.ylabel("Salary (USD)")
plt.show()


#This will help to know impact of expereince over time
plt.figure(figsize=(8, 5))
sns.boxplot(x="experience_level", y="salary_in_usd", data=df)
plt.title("Salary Based on Experience Level")
plt.xlabel("Experience Level")
plt.ylabel("Salary (USD)")
plt.show()


df = pd.read_csv("processed_salary_data.csv")


# Select relevant features for prediction
features = ["experience_level", "employment_type", "job_title", "remote_ratio", "company_location"]
target = "salary_in_usd"

# Convert categorical features into numerical values
label_encoders = {}
for col in features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Store encoders for future use

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

print("Data prepared successfully for ML model training!")


# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model Performance:")
print(f"Mean Absolute Error: {mae}")
print(f"R² Score: {r2}")

# Example: Predict salary for a new job profile
new_data = pd.DataFrame({
    "experience_level": [label_encoders["experience_level"].transform(["SE"])[0]],  # Example: "SE" (Senior)
    "employment_type": [label_encoders["employment_type"].transform(["FT"])[0]],  # Example: "FT" (Full-time)
    "job_title": [label_encoders["job_title"].transform(["Data Scientist"])[0]],
    "remote_ratio": [100],  # Fully remote job
    "company_location": [label_encoders["company_location"].transform(["US"])[0]],  # Company in US
})

# Predict salary
predicted_salary = model.predict(new_data)
print(f"Predicted Salary: ${predicted_salary[0]:,.2f}")

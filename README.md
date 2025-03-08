# 📊 Telco Customer Churn Analysis

## 📌 Overview
This project analyzes customer churn in a **telecommunications company**. The dataset contains customer demographics, service usage, and contract details to identify patterns associated with churn.

The goal is to:
- Perform **Exploratory Data Analysis (EDA)** to identify trends.
- Handle **data preprocessing** and feature engineering.
- Use **visualizations** for insights.
- Optionally, apply **machine learning models** to predict churn.

---

## 🗂️ Dataset Description
The dataset includes:
- **Customer ID**: Unique identifier.
- **Demographics**: Gender, senior citizen status, partner, and dependents.
- **Service Information**: Internet service, online security, streaming TV, etc.
- **Contract Details**: Contract type, paperless billing, payment method.
- **Churn Label**: Whether the customer left the service (`Yes` or `No`).

📌 **Data Cleaning Steps:**
- Handled missing values.
- Converted categorical variables.
- Engineered new features for analysis.

---

## ⚙️ Installation & Setup

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/yourusername/Telco-Customer-Churn.git
cd Telco-Customer-Churn
```
### **2️⃣ Install Dependencies**
Ensure you have **Python 3.x** installed, then install required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### **3️⃣ Run the Jupyter Notebook**
```bash
jupyter notebook
```
Open `Telco Customer Churn.ipynb` and execute all cells.

---

## 🔍 Exploratory Data Analysis (EDA)

### **1️⃣ Data Summary**
```python
import pandas as pd

df = pd.read_csv("Customer Churn.csv")
print(df.info())  # Dataset structure
print(df.describe())  # Statistical summary
print(df.isnull().sum())  # Check missing values
```

### **2️⃣ Churn Distribution**
```python
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))
sns.countplot(x='Churn', data=df, palette='coolwarm')
plt.title("Customer Churn Distribution")
plt.show()
```
📊 **Insight**: Helps understand the proportion of customers who churned vs. stayed.

![Churn Distribution](https://via.placeholder.com/600x400?text=Churn+Distribution)

---

### **3️⃣ Correlation Heatmap**
```python
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='Blues')
plt.title("Feature Correlation Heatmap")
plt.show()
```
📊 **Insight**: Identifies relationships between different variables.

![Heatmap](https://via.placeholder.com/600x400?text=Feature+Correlation)

---

## ✨ Feature Engineering
Some feature transformations:
- **Encoding categorical variables** (`Yes/No`, `Male/Female` → `0/1`).
- **Creating new aggregated features**.
- **Removing redundant columns**.

Example transformation:
```python
df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
df = pd.get_dummies(df, drop_first=True)  # Convert categorical to numerical
```

---

## 📈 Predicting Customer Churn (Optional)

### **1️⃣ Splitting Data for Modeling**
```python
from sklearn.model_selection import train_test_split

X = df.drop(columns=['Churn'])
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### **2️⃣ Applying a Machine Learning Model**
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
```
📊 **Insight**: This gives a baseline model to predict churn.

---

## 📊 Key Visualizations

### **1️⃣ Churn by Contract Type**
```python
plt.figure(figsize=(8,5))
sns.countplot(x='Contract', hue='Churn', data=df)
plt.title("Churn Rate by Contract Type")
plt.show()
```
📊 **Insight**: Customers with **month-to-month contracts** have a higher churn rate.

![Contract Type](https://via.placeholder.com/600x400?text=Churn+by+Contract+Type)

### **2️⃣ Monthly Charges vs. Churn**
```python
plt.figure(figsize=(8,5))
sns.boxplot(x="Churn", y="MonthlyCharges", data=df)
plt.title("Monthly Charges vs Churn")
plt.show()
```
📊 **Insight**: Higher monthly charges correlate with increased churn.

![Monthly Charges](https://via.placeholder.com/600x400?text=Monthly+Charges+vs+Churn)

---

## 🏆 Results & Insights
- **Customers with month-to-month contracts are more likely to churn.**
- **Senior citizens have a slightly higher churn rate.**
- **Paperless billing customers churn more frequently.**
- **Long-term contract customers are more loyal.**

📢 **Business Recommendation**: Offer incentives for long-term contracts to reduce churn.

---

## 🏗️ Future Improvements
- ✅ Improve feature selection for better model accuracy.
- ✅ Implement hyperparameter tuning for the ML model.
- ✅ Deploy the model via Flask or Streamlit.

---

## 🤝 Contribution & License
- Feel free to **contribute** by submitting pull requests.
- Licensed under **MIT License**.

---

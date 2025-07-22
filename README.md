# 🧠 Mental Health Prediction Dashboard

This project leverages **Machine Learning** and **Business Intelligence & Analytics (BI&A)** to predict three mental health conditions—**Anxiety, Stress, and Depression**—among university students.

A **Support Vector Machine (SVM)** model is used to power predictions. The web application allows users to assess their mental condition based on input features and displays interactive dashboards to explore both overall and individual mental health trends.

## 🚀 Features

- ✅ Predict **Anxiety**, **Stress**, or **Depression**
- 📊 Interactive BI&A dashboards
- 🔍 Model trained on real-world student data from Kaggle
- 🧩 Developed using Python and Flask

## 📁 How to Run the App

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git

2. **Navigate into the app folder**
   ```bash
   cd app

3. **Run the Flask app**
   ```bash
   python app.py

4. **Open your browser and go to**:
    ```bash
    http://localhost:5000

## 🧪 Dataset

The dataset used in this project is the **University Mental Health Dataset**, sourced from Kaggle (open-source).  
It includes responses from university students on mental health indicators such as:

- Anxiety
- Stress
- Depression
- Demographic details (e.g., gender, course, year of study)

This data was cleaned and preprocessed to be suitable for machine learning classification tasks.

---

## 🧠 Machine Learning Model

The core predictive engine of the application is a **Support Vector Machine (SVM)** classifier.

- **Target Variables**:
  - Anxiety
  - Stress
  - Depression

- **Preprocessing**:
  - One-hot encoding of categorical variables (e.g., gender, course)
  - Feature scaling where applicable

- **Evaluation Metrics**:
  - Accuracy
  - Precision
  - Recall
  - F1-Score

Each mental health condition is trained and evaluated separately using SVM to ensure tailored predictions per category.

---

## 📊 Dashboard Overview

The application includes interactive dashboards built with BI&A tools to help visualize mental health trends.

- **Overall Dashboard**:
  - Aggregated statistics across all three mental health conditions
  - Distribution charts, prevalence rates, and comparison graphs

- **Individual Dashboards**:
  - Separate dashboards for **Anxiety**, **Stress**, and **Depression**
  - Focused insights into demographics, risk factors, and prediction outcomes

These dashboards allow users and stakeholders to explore mental health data in a clear and informative way.

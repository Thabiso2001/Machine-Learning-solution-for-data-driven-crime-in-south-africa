# Machine-Learning-solution-for-data-driven-crime-in-south-africa

# 🧠 Machine Learning Crime Analysis & Forecast — Final Exam 2025

**Course:** Machine Learning  
**Exam:** Final Exam 2025  
**Student:** Manqele T.V.  
**Student Number:** 22304903  
**Lecturer:** [Lecturer’s Name]  
**Institution:** [Institution Name]  
**Date:** October 2025  

---

## 📌 Overview

This repository contains my **Machine Learning Final Exam (2025)** project titled **“Crime Analysis and Forecasting Using Machine Learning.”**  
The project explores crime data through **exploratory analysis, classification modeling, and time-series forecasting**, all deployed via an **interactive Streamlit dashboard**.  

---

## 🎯 Objectives

- Acquire and justify relevant crime data.  
- Conduct exploratory data analysis (EDA) to reveal crime patterns by category, time, and location.  
- Build and evaluate a **machine learning classification model** to predict crime types.  
- Perform **time series forecasting** to predict future crime trends with confidence intervals.  
- Present findings through a **Streamlit dashboard** for both technical and non-technical audiences.  

---

## 🧾 Dataset Description & Relevance

**Dataset Name:** Crime Incidents Dataset  
**Source:** [Download Dataset (GitHub Link)](https://github.com/ManqeleTV/Crime-Analysis-Exam2025/blob/main/crime_data.csv)

| Column | Description |
|---------|-------------|
| `date` | Date/time of recorded crime incident |
| `category` | Type of crime (e.g., Theft, Assault, Burglary) |
| `location` | Area or precinct name |
| `latitude` | Latitude coordinate of the incident |
| `longitude` | Longitude coordinate of the incident |
| Other features | Contextual or engineered predictors used for classification |

### 🔹 Relevance Justification
This dataset is suitable for analyzing **temporal**, **geographical**, and **categorical** crime patterns.  
It supports modeling to predict future crime incidents and helps stakeholders understand high-risk areas and times for better resource allocation.

---

## 🧮 Methods & Workflow

### 1️⃣ Data Preprocessing
- Converted date columns to proper datetime objects.  
- Removed duplicates and handled missing values.  
- Encoded categorical variables and normalized numerical features.  
- Filtered data based on category, location, and time period.  

### 2️⃣ Exploratory Data Analysis (EDA)
- Visualized monthly/daily crime counts.  
- Identified top crime categories and hotspot locations.  
- Generated correlation plots and heatmaps.  
- Displayed incident locations on an interactive map.  

### 3️⃣ Classification Modeling
- **Algorithm:** Random Forest Classifier (200 trees).  
- **Target Variable:** `category`.  
- **Features:** Selected by user (e.g., location, time, and engineered variables).  
- **Metrics:** Accuracy, Precision, Recall, F1-score, ROC-AUC.  
- **Visuals:** Confusion Matrix, Classification Report, Feature Importance chart.  

### 4️⃣ Time-Series Forecasting
- **Model Used:** Exponential Smoothing (Holt-Winters).  
- **Forecast Horizon:** Configurable (default 30 days).  
- **Output:** Future trend predictions with upper and lower confidence intervals.  

---

## 📊 Streamlit Dashboard

The **Streamlit Dashboard** allows users to:
- Filter crimes by **category**, **location**, and **date range**.  
- Visualize EDA plots (time trends, bar charts, heatmaps, and maps).  
- Run classification and view model performance metrics.  
- Generate forecasts with confidence intervals.  
- Read summaries for **both technical and non-technical audiences**.  

---


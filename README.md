# 📊 **Candidate Joining Prediction in Talent Acquisition**

![Talent Acquisition](images/Designer1.png)

Welcome to the **Candidate Joining Prediction in Talent Acquisition** project! 👥 This repository highlights our efforts to address a critical challenge in the talent acquisition process—predicting whether candidates will join a company after accepting a job offer. Accurately forecasting candidate behavior allows HR teams to mitigate the risk of attrition, optimize recruitment strategies, and ultimately save valuable resources.

## 🔍 **Project Overview**

HRWorks Pvt Ltd, a talent acquisition solutions provider, has faced a recurring issue where a significant number of candidates do not join the company after accepting job offers. This project aims to build a predictive model that can classify whether a candidate is likely to join or not, providing valuable insights to HR teams and reducing the financial and operational impact of attrition.

## 🛠️ **Project Approach**

### 1. **Exploratory Data Analysis (EDA)**

- 📊 Analyzed the dataset to uncover key trends and relationships between variables.
- 🧹 Identified and handled missing values, outliers, and inconsistencies in the data.
- 🔍 Conducted correlation analysis and feature engineering to enhance model performance.

### 2. **Data Preprocessing**

- 🔄 Scaled and encoded features to prepare the data for model training.
- 🧪 Split the dataset into training and testing sets to ensure unbiased model evaluation.

### 3. **Model Training**

- 🤖 Experimented with multiple classification algorithms including Logistic Regression, Random Forest, and XGBoost.
- 🧠 Performed hyperparameter tuning to optimize model performance.

### 4. **Model Evaluation**

- 🏅 Assessed models using key metrics such as Accuracy, ROC AUC, and F1 Score.
- ✅ Selected the best-performing models based on both training and testing performance.

### 5. **Feature Importance**

- 🌟 Conducted feature importance analysis to identify key factors influencing candidate decisions.
- 🎯 Focused on the most impactful features to refine the model and improve interpretability.

### 6. **Model Deployment**

- 🚀 Deployed the final model to assist HR teams in making data-driven decisions.
- 📊 Integrated the model into the recruitment process to provide real-time predictions.

## 🌟 **Key Highlights**

- **Model Performance**: Our tuned XGBoost model achieved a testing accuracy of 82.30%, with an ROC AUC of 0.749 and an F1 score of 0.349. This model provides reliable predictions that can significantly aid in candidate selection.
- **Impactful Features**: Factors such as interview performance, job role, and previous experience were identified as key drivers of candidate decisions.
- **Actionable Insights**: By predicting the likelihood of a candidate joining, HR teams can focus efforts on the most promising candidates, reducing attrition and optimizing the recruitment process.

## 📁 **Repository Structure**

- `data/`: Contains the raw and processed data used in the analysis.
- `models/`: Includes saved models and scaling objects for deployment.
- `notebooks/`: Jupyter notebooks detailing the EDA, model training, and evaluation.
- `scripts/`: Python scripts for data preprocessing, feature engineering, and model training.
- `README.md`: This document outlining the project.

## 🚀 **Getting Started**

To get started with this project, clone the repository and follow the instructions in the setup guide. Ensure that all dependencies are installed and the data is correctly placed in the `data/` directory.

---

Feel free to explore the code and contribute to the project! Together, we can further improve the recruitment process and reduce candidate attrition. 🎯

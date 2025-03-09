# Titanic Classification Project

## 📌 Overview
This project aims to predict passenger survival on the Titanic using machine learning techniques. 
The dataset contains demographic and travel details of passengers, which are used to train a classification model.

## 📂 Dataset
The dataset used in this project is the well-known **Titanic dataset**, which includes:
- **Passenger information** (e.g., Name, Age, Sex, Ticket Class)
- **Travel details** (e.g., Embarked location, Fare, Cabin)
- **Survival status** (0 = Did not survive, 1 = Survived)

## 🛠️ Setup and Installation

### 1️⃣ Prerequisites
Ensure you have **Python 3.x** installed along with the following libraries:
```bash
pip install pandas numpy seaborn matplotlib scikit-learn
```

### 2️⃣ Clone Repository
```bash
git clone <repository_link>
cd Titanic-Classification
```

### 3️⃣ Running the Jupyter Notebook
```bash
jupyter notebook
```
Open the notebook **Titanic Classification.ipynb** and run the cells step by step.

## 🔍 Exploratory Data Analysis (EDA)
Performed data exploration using:
- **Missing Value Analysis**
- **Feature Correlation Analysis**
- **Distribution of Passenger Attributes**
- **Survival Rate based on Features**

## 📊 Data Preprocessing
The following preprocessing steps were applied:
- **Handling Missing Values** (Imputation of missing Age and Embarked values)
- **Feature Encoding** (Using `pd.get_dummies()` for categorical variables)
- **Feature Scaling** (Normalization of Fare and Age)
- **Splitting Data** into Training and Test Sets

## 🏗️ Model Training
model trained various classification model:
1. **Logistic Regression**

The best-performing model was selected based on **accuracy, precision, recall, and F1-score.**

## 🎯 Model Evaluation
The trained model achieved the following performance metrics:
- **Accuracy:** 1.00
- **ROC AUC Score:** 1.0
- **Classification Report:**
  ```
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        50
           1       1.00      1.00      1.00        34

    accuracy                           1.00        84
   macro avg       1.00      1.00      1.00        84
weighted avg       1.00      1.00      1.00        84
  ```

## ⚠️ Potential Overfitting Check
Since the model achieved **100% accuracy and ROC AUC = 1.0**, we checked for:
- **Data Leakage:** Ensured no direct survival information was used.
- **Overfitting:** Used **cross-validation** and checked feature importance.

## 🚀 Future Improvements
- Use **hyperparameter tuning** to improve model generalization.
- Experiment with **ensemble methods**.
- Deploy the model using **Flask or Streamlit**.

## 🏆 Conclusion
This project successfully classifies Titanic passengers based on survival probability.
However, the perfect scores suggest further investigation is needed to ensure generalization.

## 🤝 Contributing
Feel free to fork and contribute to this project!

## 📜 License
This project is open-source under the **MIT License**.

## Author
Sherly Yaqoob 

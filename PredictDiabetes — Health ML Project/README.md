**Diabetes Prediction Using Machine Learning**

This project uses the CDC BRFSS 2015 balanced diabetes dataset to predict diabetes through multiple machine-learning models and a neural network. It includes full data cleaning, exploratory analysis, feature engineering, model training, and performance comparison.

** Project Overview**

This project analyzes health-indicator data—BMI, smoking, physical activity, blood pressure, cholesterol, and more—to determine whether a person is likely to have diabetes.

**The workflow includes:**

Data cleaning & preprocessing

Exploratory Data Analysis (EDA) with visualizations

**Feature engineering**

Training multiple machine-learning models

Building a neural network using TensorFlow

Comparing accuracy across all models

** Dataset**

Source: CDC BRFSS 2015 (balanced 50/50 split of diabetic vs non-diabetic cases)

Important columns include:

Diabetes_Status (target variable)

BMI, HighBP, HighChol

Smoking, AlcoholConsumption

HeartDisease, Stroke

Physical Activity

Fruits / Veggies consumption

Age, Sex, Race

** Technologies Used**

Python 3.8+

Pandas, NumPy

Matplotlib, Seaborn

Scikit-learn

TensorFlow / Keras

**Exploratory Data Analysis**

Includes:

Gender distribution pie charts

Age-range bar charts

Diabetes case counts

Health indicator heatmaps

Crosstabs for:

Smoking vs Diabetes

BMI categories

High Blood Pressure

Physical Activity

Alcohol consumption

** Models Implemented**
Traditional Machine Learning Models

K-Nearest Neighbors (KNN)

Logistic Regression

Decision Tree

Random Forest

AdaBoost (default + SAMME)

Gradient Boosting

Deep Learning

Keras Sequential neural network:

Dense(128, relu)

Dense(64, relu)

Dense(10, softmax)

**Model Evaluation**

All models use:

70% training data  
30% testing data


**Evaluation includes:**

Accuracy Score

Predictions vs True Labels

Comparison of all model performances

** How to Run the Project**
1. Clone the repository
git clone https://github.com/yourusername/diabetes-ml-project.git
cd diabetes-ml-project

2. Install dependencies
pip install -r requirements.txt

3. Run the notebook

Open Jupyter and run the .ipynb file.

** Repository Structure**
│
├── diabetes-ml.ipynb       # Main notebook
├── sample_plot.png         # Example visualization
├── pie.jpeg                # Gender pie chart
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies

**Contributing**

Pull requests are welcome.
Feel free to open issues or suggest improvements.

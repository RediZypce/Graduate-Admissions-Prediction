# Graduate Admissions Prediction
![image](https://github.com/RediZypce/Graduate-Admissions-Prediction/assets/109640560/62b972fb-f9be-46a9-aa26-bb5a5bfdf92a)

# Overview
This project is aimed at building a deep learning model to predict the likelihood of graduate admission based on various applicant profile features. The dataset used for this project is "admissions_data.csv," which contains information about graduate applicants and their attributes. The model is developed using deep learning techniques and is intended to provide a predictive tool for evaluating the chances of admission for graduate applicants. [Get Started](Graduate_Admissions_Prediction.ipynb)

# Dataset
The dataset used in this project is named ["admissions_data.csv."]("admissions_data.csv.") It is a CSV (Comma-Separated Values) file containing information about graduate applicants, including features such as GRE scores, TOEFL scores, university ratings, statement of purpose strength (SOP), letter of recommendation strength (LOR), undergraduate GPA (CGPA), research experience, and the target variable, which is the "Chance of Admit."

# Data Features
The dataset includes the following columns:

* Serial No.: An index for each row (ranging from 1 to 500).
* GRE Score: GRE test scores (scaled out of 340).
* TOEFL Score: TOEFL test scores (scaled out of 120).
* University Rating: Evaluated university rating (ranging from 1 to 5).
* SOP: Strength of the Statement of Purpose (ranging from 1 to 5).
* LOR: Strength of the Letter of Recommendation (ranging from 1 to 5).
* CGPA: Undergraduate GPA (scaled out of 10).
* Research: Binary indicator (0 or 1) for research experience.
* Chance of Admit: The target variable, indicating the applicant's chance of being admitted (ranging from 0 to 1).

# Project Objectives
The primary objectives of this project are as follows:

__Model Development:__ Build a deep learning model for predicting the probability of graduate admission based on applicant profiles.

__Feature Engineering:__ Select and preprocess relevant features from the dataset to train the model effectively.

__Model Evaluation:__ Assess the performance of the trained model using appropriate evaluation metrics, such as Mean Absolute Error (MAE) and R-squared (R2) score.

__Early Stopping:__ Implement early stopping as a regularization technique to prevent overfitting during training.

__Visualizations:__ Create plots to visualize the model's training progress, changes in MAE and loss, and feature importances.

# Dependencies
This project uses the following libraries and tools:

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* TensorFlow
* Keras
* Scikit-Learn
You can import the above libraries either from the [script.py](script.py) or through the [Jupyter Notebook](Graduate_Admissions_Prediction.ipynb).

# Usage
Follow the instructions in the Jupyter Notebook to load the dataset, preprocess the data, build the deep learning model, and evaluate its performance.

# Visualizations
In addition to model training and evaluation, the project includes exploratory data visualizations to better understand the dataset. You can find these visualizations in the Jupyter Notebook.

# Contact Me
Feel free to contact me on [X](https://twitter.com/RediZypce)

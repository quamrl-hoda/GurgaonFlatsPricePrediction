

#  Flats Price Prediction using Machine Learning

This project builds a **machine learning model** to predict the price of flats/apartments based on important features such as **location, area, number of rooms, and amenities**.
It demonstrates a full end-to-end ML workflow — from data preprocessing to model deployment.

---

##  **Project Overview**

Accurate price prediction is one of the most important applications of machine learning in the real-estate domain.
In this project, we prepare a clean dataset, explore relationships, engineer important features, train multiple ML models, and select the best one for prediction.

The goal is simple:
**Build a model that can reliably predict flat prices based on real-world property characteristics.**

---

##  **Features**

*  Complete data preprocessing
*  Handling missing values & outliers
*  Exploratory Data Analysis (EDA)
*  Feature engineering for real-estate attributes
*  Model training & comparison
*  Visualizations of price distribution and trends
*  Saved trained model for predictions
*  Ready-to-use prediction script/notebook

---

##  **Technologies Used**

* Python
* NumPy, Pandas
* Matplotlib, Seaborn
* Scikit-learn
* Jupyter Notebook

---

##  **Project Structure**

```
FLATSPREDICTION/
│
├── datasets/
│   ├── cosine_sim3.pkl
│   ├── data_viz1.csv
│   ├── df.pkl
│   ├── feature_text.pkl
│   ├── location_df.pkl
│   └── pipeline.pkl
│
├── models/
│   ├── __init__.py
│   ├── price_predictor.py
│   ├── recommender.py
│   └── visualizer.py
│
├── myenv/                      # Virtual environment (optional to push)
│
├── static/
│   ├── script.js
│   └── style.css
│
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── predict.html
│   ├── recommend.html
│   └── visualizer.html
│
├── .gitignore
├── main.py                     # Main Flask app / backend controller
├── README.md
└── requirements.txt

---

##  **Models Used**

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor
* Gradient Boosting / XGBoost (optional)

The best-performing model is selected based on **R² score**, **MAE**, and **RMSE**.

---

##  **How to Run the Project**

### 1️ Clone the repository

```bash
git clone https://github.com/your-username/flats-price-prediction.git
cd flats-price-prediction
```

### 2 Install dependencies

```bash
pip install -r requirements.txt
```

### 3 Run the notebook

```bash
jupyter notebook
```

###  Make a prediction

```bash
python src/predict.py
```

---

##  **Results**

The model delivers strong performance on the test dataset and provides meaningful insights into features that impact flat pricing the most (e.g., location, area, rooms).

---

##  **Contributions**

Feel free to open issues or submit pull requests for improvements, new models, or better visualizations.

---

##  **Contact**

If you'd like to connect or collaborate:

**GitHub:** github.com/quamrl-hoda
**LinkedIn:** linkedin.com/in/qumarl-hoda

---

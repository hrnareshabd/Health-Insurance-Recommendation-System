# 🏥 Health Insurance Recommendation System using Machine Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Google Colab](https://img.shields.io/badge/Google-Colab-F9AB00?logo=googlecolab)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikitlearn)
![CatBoost](https://img.shields.io/badge/CatBoost-Best%20Model-yellow)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

# Health Insurance Recommendation System

### End-to-End Machine Learning Project for Insurance Charge Prediction & Recommendation

*A complete Machine Learning pipeline that predicts whether a customer is likely to incur **High** or **Low** insurance charges and provides a recommendation based on the prediction.*

</div>

---

# 📌 Project Overview

Choosing the right health insurance plan is often difficult because medical expenses depend on several factors such as **age, BMI, smoking habits, number of children, and geographic region**.

This project develops a **Machine Learning-based Health Insurance Recommendation System** that predicts insurance charge categories using customer demographic and health information.

The project demonstrates an end-to-end Machine Learning workflow including:

* Data Collection
* Data Cleaning
* Exploratory Data Analysis
* Feature Engineering
* Hyperparameter Tuning
* Model Training
* Model Evaluation
* Prediction System

---

# 🚀 Project Highlights

* ✅ Built using **Python** and **Google Colab**
* ✅ Dataset containing **1,338 customer records**
* ✅ Compared **7 Machine Learning Algorithms**
* ✅ Applied Hyperparameter Tuning
* ✅ Selected the Best Performing Model
* ✅ Built an Insurance Recommendation Pipeline
* ✅ Tested on Multiple Realistic User Profiles

---

# 📊 Dataset Information

| Property        | Value                         |
| --------------- | ----------------------------- |
| Dataset Size    | **1,338 Records**             |
| Features        | **6 Input Features + Target** |
| Missing Values  | **None**                      |
| Target Variable | Insurance Charges             |
| Binary Target   | High Charges / Low Charges    |

### Features

* Age
* Sex
* BMI
* Children
* Smoker
* Region

Target Variable

* Insurance Charges

For model training, the target was converted into a binary classification problem:

* **High Charges**
* **Low Charges**

using the median insurance charge (**$9,382.03**) as the threshold.

---

# ⚙ Machine Learning Workflow

```text
Dataset
   │
   ▼
Data Cleaning
   │
   ▼
Exploratory Data Analysis
   │
   ▼
Feature Encoding
   │
   ▼
Feature Scaling
   │
   ▼
Train-Test Split
   │
   ▼
Hyperparameter Tuning
   │
   ▼
Train 7 ML Models
   │
   ▼
Model Comparison
   │
   ▼
Best Model Selection
   │
   ▼
Prediction & Recommendation
```

---

# 🤖 Machine Learning Models

The following supervised learning algorithms were implemented and compared.

| Model               |   Accuracy |   F1 Score |    ROC-AUC |
| ------------------- | ---------: | ---------: | ---------: |
| Logistic Regression |     91.04% |     0.9111 |     0.9433 |
| Random Forest       |     94.03% |     0.9380 |     0.9480 |
| XGBoost             |     93.66% |     0.9344 |     0.9532 |
| LightGBM            |     93.66% |     0.9344 |     0.9488 |
| ⭐ CatBoost          | **94.78%** | **0.9453** | **0.9439** |
| SVM                 |     92.91% |     0.9278 | **0.9659** |
| KNN                 |     91.42% |     0.9125 |     0.9503 |

---

# 🏆 Best Model

After comparing all seven models, **CatBoost** achieved the best overall performance.

| Metric                 |               Score |
| ---------------------- | ------------------: |
| Model                  |          ⭐ CatBoost |
| Accuracy               |          **94.78%** |
| F1 Score               |          **0.9453** |
| ROC-AUC                |          **0.9439** |
| Cross Validation Score | **0.9297 ± 0.0124** |

---

# 🔧 Hyperparameter Tuning

Randomized Search was used to optimize each model.

Example of optimized parameters:

| Model               | Best Parameters                |
| ------------------- | ------------------------------ |
| Logistic Regression | C = 1                          |
| Random Forest       | 200 Trees, Depth = 15          |
| XGBoost             | 200 Trees, Depth = 6           |
| LightGBM            | 200 Trees, 31 Leaves           |
| CatBoost            | Depth = 6, Learning Rate = 0.1 |
| SVM                 | C = 1, Gamma = scale           |
| KNN                 | k = 5                          |

---

# 📈 Model Evaluation

Models were evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Cross Validation

This provides a comprehensive comparison rather than relying solely on accuracy.

---

# 🧪 Recommendation System Testing

The recommendation pipeline was evaluated using **6 realistic customer profiles**.

| Test Cases              | Result    |
| ----------------------- | --------- |
| Total Test Cases        | 6         |
| Correct Predictions     | 5         |
| Recommendation Accuracy | **83.3%** |

### Example Test Cases

* Young Healthy Professional
* Middle-aged Smoker
* Senior Citizen
* Young Family
* Overweight Smoker
* Healthy Senior

The system correctly classified **5 out of 6** customer profiles.

---

# 📷 Project Screenshots

In the **Project Output** Folder

Added screenshots in the **Project Output** Folder.

```text
images/

Dataset.png

Feature Correlation Matrix.png

COMPREHENSIVE MODEL COMPARISON.png

BoxPlot Grouped by Recommendation.png

Model Comparison.png

LIME Explaination.png

TOP 5 Model Comparison.png

FEATURE IMPORTANCE ANALYSIS.png

HyperParameter Tuning For 7 Model.png

HyperParameter Tuning For 7 Model 2.png

Test Summary.ong

Final Recomendation.png
```


```markdown
## Model Comparison

![Models](Project Output/TOP 5 Model Comparison.png)
```

---

# 🛠 Technologies Used

* Python
* Google Colab
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* CatBoost
* XGBoost
* LightGBM
* Joblib

---

# ▶ Installation

```bash
git clone https://github.com/hrnareshabd/Health-Insurance-Recommendation.git

cd Health-Insurance-Recommendation

pip install -r requirements.txt

jupyter notebook
```

Or simply open the notebook using **Google Colab**.

---

# 📂 Project Structure

```text
Health-Insurance-Recommendation

│── notebook/
│      Health_Insurance_Recommendation.ipynb

│── data/
│      insurance.csv

│── images/

│── requirements.txt

│── README.md

│── LICENSE
```

---

# 🔮 Future Improvements

* Deploy using Streamlit
* Build a REST API using FastAPI
* Add SHAP Explainable AI
* Dockerize the project
* Deploy on AWS or Azure
* Add CI/CD with GitHub Actions

---

# 👨‍💻 Author

**Naresh Hosahalli Rudresh**

🎓 MSc Data Science

📍 Germany

* GitHub: https://github.com/hrnareshabd
* LinkedIn: https://www.linkedin.com/in/naresh-h-r/

---

# ⭐ Support

If you found this project useful,

⭐ Star this repository

🍴 Fork it

📢 Share it with others

---

# 📜 License

This project is licensed under the MIT License.

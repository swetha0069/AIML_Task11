# AIML_Task11
# AI & ML Internship | Task 11: Breast Cancer Classification

## 📌 Project Overview
This project implements a **Support Vector Machine (SVM)** classifier to predict breast cancer malignancy. The goal was to build a robust pipeline that includes data normalization, hyperparameter tuning, and comprehensive evaluation metrics.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Libraries:** Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn
* **Model:** Support Vector Machine (SVM)
* **Tools:** GridSearchCV (Tuning), Joblib (Model Persistence)

## 🚀 Implementation Workflow
1.  **Data Acquisition:** Loaded the `sklearn.datasets.load_breast_cancer()` dataset.
2.  **Preprocessing:** Applied `StandardScaler` to normalize feature ranges.
3.  **Baseline Training:** Trained an initial SVM with a `linear` kernel to establish a performance floor.
4.  **Optimization:** Performed a Grid Search to find the optimal balance between `C` (penalty) and `gamma` (kernel coefficient).
5.  **Evaluation:** Analyzed results using a Confusion Matrix, Classification Report (Precision/Recall), and AUC-ROC curve.

## 📊 Key Results
* **Best Parameters:** `kernel: 'rbf'`, `C: 1.0`, `gamma: 'scale'` (vary based on your grid search results).
* **Accuracy:** ~96-98% (consistent across train/test splits).
* **AUC Score:** High area under the curve indicates strong model discriminative power between Malignant and Benign classes.



## 📂 Deliverables
* `notebook.ipynb`: Full implementation and visualization.
* `svm_model.pkl`: The final tuned SVM model.
* `scaler.pkl`: The StandardScaler object used for feature normalization.

## 💡 Interview Concepts Covered
* **SVM Margin:** Maximizing the distance between the hyperplane and support vectors.
* **Kernel Trick:** Transforming data into higher dimensions to handle non-linearity.
* **C Parameter:** Trade-off between misclassification and decision boundary simplicity.
* **Gamma:** Influence range of a single training example.

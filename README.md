# KNN
Here is a comprehensive, production-ready `README.md` file based directly on the machine learning workflow in your Jupyter Notebook.

---

# K-Nearest Neighbors (KNN) Region Classification

This repository contains a complete machine learning pipeline that uses the **K-Nearest Neighbors (KNN)** algorithm to classify spatial coordinates into one of three distinct regional categories (`Red`, `Blue`, or `Yellow`).

The pipeline handles data preprocessing, hyperparameter optimization, model evaluation, and inference on unseen data.

---

## 📋 Features

* **Data Exploration:** Automated structural and missing value verification.
* **Feature Engineering:** Stratified data splitting and data normalization using standard scaling ($Z$-score normalization).
* **Hyperparameter Tuning:** Empirical evaluation of $k$-values ranging from 1 to 30 to optimize classification accuracy.
* **Comprehensive Evaluation:** Generates precision, recall, and $F_1$-score matrices alongside a multi-class confusion matrix.
* **Probability Inference:** Computes explicit class membership probabilities for new data points.

---

## 🛠️ Requirements & Installation

To run this notebook locally, ensure you have Python 3 installed along with the following standard data science libraries:

```bash
pip install pandas scikit-learn matplotlib

```

---

## 📊 Dataset Profile

The model processes a spatial layout file named `overlapping_dataset_200.csv`.

* **Total Samples:** 200 rows
* **Features:** * `x` (float64): Horizontal spatial coordinate
* `y` (float64): Vertical spatial coordinate


* **Target variable:** `Region` (object): Three-class categorical label (`Red`, `Blue`, `Yellow`)

---

## 🚀 Pipeline Workflow

### 1. Data Prep & Scaling

The dataset is cleanly split into training and testing sets ($80/20$ split), maintaining class proportions via stratification. Features are scaled using `StandardScaler` to prevent distance distortions during neighbor calculations:

$$Z = \frac{x - \mu}{\sigma}$$

### 2. Hyperparameter Tuning ($k$-Value Optimization)

The notebook evaluates the model across a spectrum of neighbor counts ($k \in [1, 30]$) and plots **Accuracy vs. K** to pinpoint the ideal balance between underfitting and overfitting.

### 3. Model Evaluation

A final model is built using the optimized hyperparameter **$k = 15$**, achieving a solid baseline performance.

#### Performance Metrics:

* **Overall Test Accuracy:** `82.5%` (0.825)
* **Classification Breakdown:**

| Region Class | Precision | Recall | F1-Score | Support |
| --- | --- | --- | --- | --- |
| **Blue** | 0.83 | 0.71 | 0.77 | 14 |
| **Red** | 0.75 | 0.86 | 0.80 | 14 |
| **Yellow** | 0.92 | 0.92 | 0.92 | 12 |

#### Confusion Matrix:

```text
[[10  3  1]   <- True Blue
 [ 2 12  0]   <- True Red
 [ 0  1 11]]  <- True Yellow

```

---

## 🔮 Inference Example

The pipeline includes a deployment-ready snippet to test new, unlabelled coordinates. For instance, predicting the region for coordinates $(x=5, y=2)$:

```python
new_data = pd.DataFrame({'x': [5], 'y': [2]})
new_data_scaled = scaler.transform(new_data)

prediction = model.predict(new_data_scaled)
probabilities = model.predict_proba(new_data_scaled)

```

**Output Results:**

* **Predicted Region:** `Red`
* **Class Probabilities:** * Blue: `33.3%`
* Red: `66.7%`
* Yellow: `0.0%`

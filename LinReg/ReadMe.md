
# Linear Regression from Scratch

This project demonstrates a **Linear Regression model implemented entirely from scratch** in Python. The implementation focuses on understanding the **mathematical foundation** of regression : including feature scaling, loss computation, and gradient descent without relying on high-level machine learning libraries.

---

##  Overview

The model predicts **housing prices** based on attributes such as area, number of bathrooms, and other categorical factors. The workflow compares baseline models using scikit-learn (`LinearRegression`, `Ridge`, `Lasso`) with a fully **from-scratch implementation** of gradient descent–based Linear Regression.

---

##  Key Features

* **Custom Linear Regression implementation**

  * Implements **gradient descent** manually for parameter updates.
  * Includes **mean, standard deviation, and scaling functions** coded from first principles.

* **Feature Scaling**

  * Manual implementation of normalization and standardization for stable gradient descent.

* **Loss Function**

  * Mean Squared Error (MSE) derived and coded manually for optimization feedback.

* **Gradient Descent Optimization**

  * Custom update rule for slope (`m`) and intercept (`b`):
    [
    m_{new} = m - \alpha \frac{2}{N} \sum x(y - (mx + b)), \quad
    b_{new} = b - \alpha \frac{2}{N} \sum (y - (mx + b))
    ]
  * Iterative visualization of loss convergence.

* **Baseline Comparison**

  * Ridge, Lasso, and standard Linear Regression (via scikit-learn) for benchmarking.

---

## Project Structure

```
linregfs.py
│
├── Data Preprocessing
│   ├── Loads dataset from Google Drive (Housing.csv)
│   ├── One-hot encodes categorical features
│   └── Scales numerical variables manually and via StandardScaler
│
├── Baseline Models
│   ├── Linear Regression
│   ├── Ridge Regression (with RidgeCV)
│   └── Lasso Regression (with LassoCV)
│
└── From Scratch Implementation
    ├── Manual mean/std computation
    ├── Custom feature and target scaling
    ├── Gradient descent and loss functions
    └── Iterative optimization and loss tracking
```

---

##  Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/ml-models-from-scratch.git
cd ml-models-from-scratch
```

Ensure dependencies are installed:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn plotly keras
```

---

## Usage

1. Open the script or notebook in Google Colab or your local environment.
2. Modify the path to your dataset:

   ```python
   file_path = '/content/drive/MyDrive/Housing.csv'
   ```
3. Run the script to:

   * Preprocess and encode features.
   * Train baseline regression models.
   * Execute the **from-scratch Linear Regression** gradient descent routine.

---

##  Example Output

```
Starting gradient descent at Loss = 0.9852
Iteration 0: Loss = 0.9637
Iteration 100: Loss = 0.3125
...
Scaled Optimal m (Slope for area_scaled): 0.8341
Scaled Optimal b (Intercept/Bias): -0.0217
Final Scaled Loss (MSE): 0.0412
```



---

Would you like me to format this README to visually match your “ML Models From Scratch” repo (e.g., using consistent emoji headers, colorized code blocks, and unified Markdown style)?

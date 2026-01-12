🧠 Handwritten Digit Recognition using Classical Machine Learning
📌 Project Overview

This project implements an end-to-end machine learning pipeline to classify handwritten digits (0–9) using the MNIST dataset.
The focus is on classical machine learning algorithms, avoiding neural networks, to demonstrate strong fundamentals in data preprocessing, model selection, evaluation, and ensemble learning.

The project was completed as part of a company assignment to evaluate practical ML understanding, experimentation, and analysis skills.

🎯 Objectives

Perform Exploratory Data Analysis (EDA) on image data stored in CSV format

Preprocess and normalize high-dimensional pixel data

Reduce dimensionality using Principal Component Analysis (PCA)

Train and evaluate multiple classical ML models

Tune hyperparameters using cross-validation

Analyze model errors and misclassifications

Improve performance using boosting and ensemble learning

📂 Dataset

Dataset: MNIST (CSV format)

Training samples: 60,000

Test samples: 10,000

Features: 784 pixel values (28×28 grayscale images)

Labels: Digits from 0 to 9

The dataset was provided with a predefined train–test split, and no further splitting was performed to avoid data leakage.

🧪 Exploratory Data Analysis (EDA)

EDA was performed on the training data and included:

Dataset shape and integrity checks (missing values, duplicates)

Class distribution analysis

Visualization of random digit samples

Pixel intensity distribution analysis

Mean image visualization for each digit class

These steps helped in understanding digit patterns and common ambiguities between similar digits.

⚙️ Data Preprocessing

Normalization: Pixel values scaled to the range [0, 1]

Dimensionality Reduction:

PCA applied with 95% variance retention

Reduced feature space significantly, improving training speed and stability

🤖 Models Implemented
1. K-Nearest Neighbors (KNN)

PCA applied before training

k tuned using a loop over odd values

Distance-based weighting used for final model

Pros: Simple, strong baseline
Cons: Slow inference, high memory usage

2. Support Vector Machine (SVM – RBF Kernel)

PCA applied to reduce dimensionality

Hyperparameters (C, gamma) tuned using GridSearchCV

Achieved the highest individual model accuracy

Pros: Excellent generalization
Cons: Computationally expensive

3. Decision Tree

Trained on normalized data

Hyperparameters tuned using GridSearchCV to control overfitting

Pros: Fast, interpretable
Cons: Lower accuracy on high-dimensional image data

📈 Model Evaluation

Models were evaluated using:

Accuracy

Confusion Matrix

Precision, Recall, and F1-score

Per-class accuracy

Visualization of misclassified digits

This detailed evaluation provided insights into systematic errors and digit confusions (e.g., 4 vs 9, 3 vs 5).

🚀 Boosting and Ensemble Learning
Boosting

AdaBoost applied to shallow Decision Trees (max_depth = 2)

Improved performance of Decision Tree as a weak learner

Ensemble Learning

Voting Classifier (Hard Voting) combining:

KNN

SVM

Boosted Decision Tree

The ensemble model demonstrated improved robustness and overall performance compared to individual models.

📊 Results Summary
Model	Accuracy (Approx.)
KNN	96–97%
SVM (RBF)	97–98%
Decision Tree	85–88%
Boosted Decision Tree	90–92%
Ensemble Model	~98%
🧠 Key Challenges & Solutions
Challenge	Solution
High dimensionality	PCA
Slow training	PCA + subset tuning
Overfitting (DT)	Depth & split constraints
Hyperparameter selection	GridSearchCV
Model instability	Ensemble learning
🛠️ Technologies Used

Python

NumPy, Pandas

Matplotlib, Seaborn

Scikit-learn

📌 Conclusion

This project demonstrates that classical machine learning techniques, when combined with proper preprocessing, dimensionality reduction, and ensemble methods, can achieve strong performance on image classification tasks.
Among individual models, SVM performed best, while the ensemble model provided the most robust results.

🔮 Future Improvements

Soft voting and stacking ensembles

Random Forest comparison

Time and memory complexity analysis

Neural network baseline comparison (optional)

👤 Author

Chetan
Machine Learning & AI Enthusiast

# FUTURE_ML_02 – Support Ticket Classification

> **Future Interns – Machine Learning Internship | Task 2**

## 📌 Overview

This project builds an NLP system that automatically classifies customer support tickets into categories such as Hardware, HR Support, Access, and Storage. Routing tickets to the right team automatically helps support teams respond faster and more consistently. The model uses TF-IDF text features with a Logistic Regression classifier and reaches **84.9% accuracy** on unseen tickets.

---

## 🎯 Problem Statement

Support teams receive thousands of tickets, and sorting them by hand is slow, repetitive, and error-prone. Tickets that land with the wrong team get delayed, which hurts response times and customer satisfaction.

**Task:** Build a system that reads the text of a support ticket and automatically predicts its category, then evaluate how well it performs.

---

## 🚀 Features

- Data Cleaning (null handling, type conversion, duplicate removal)
- Text Preprocessing (lowercasing, special character removal, whitespace normalization)
- TF-IDF Feature Extraction with stop-word removal
- Multi-class Ticket Category Classification (8 categories)
- Stratified Train/Test Split
- Model Evaluation (Accuracy, Precision, Recall, F1, Confusion Matrix)
- Predicting the category of a brand-new ticket

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook

---

## 📊 Dataset

The dataset (`all_tickets_processed_improved_v3.csv`) contains **47,837 support tickets** with 2 columns and no missing values.

| Column | Description |
|---|---|
| Document | Pre-processed text of the support ticket |
| Topic_group | Ticket category (target) |

**Category distribution**

| Category | Tickets |
|---|---:|
| Hardware | 13,617 |
| HR Support | 10,915 |
| Access | 7,125 |
| Miscellaneous | 7,060 |
| Storage | 2,777 |
| Purchase | 2,464 |
| Internal Project | 2,119 |
| Administrative rights | 1,760 |

The classes are imbalanced: Hardware has about 8x more tickets than Administrative rights.

---

## 🧹 Methodology

1. **Data cleaning** – dropped null rows, converted columns to text, removed duplicates (47,837 rows remained).
2. **Text cleaning** – lowercased text, replaced special characters with spaces, and collapsed extra whitespace.
3. **Train/test split** – 80/20 split (**38,269 train / 9,568 test**) with `stratify=y` so every category keeps the same proportion in both sets.
4. **Pipeline** – a scikit-learn `Pipeline` combining:
   - `TfidfVectorizer(stop_words="english", max_features=5000)`
   - `LogisticRegression(max_iter=1000)`
5. **Evaluation** – accuracy, per-class classification report, and confusion matrix on the test set.

---

## 📈 Model Performance

**Overall accuracy: 84.88%** | Macro F1: 0.85 | Weighted F1: 0.85

| Category | Precision | Recall | F1 Score | Support |
|---|---:|---:|---:|---:|
| Access | 0.91 | 0.87 | 0.89 | 1,425 |
| Administrative rights | 0.89 | 0.65 | 0.75 | 352 |
| HR Support | 0.86 | 0.85 | 0.85 | 2,183 |
| Hardware | 0.78 | 0.89 | 0.83 | 2,724 |
| Internal Project | 0.91 | 0.80 | 0.85 | 424 |
| Miscellaneous | 0.82 | 0.82 | 0.82 | 1,412 |
| Purchase | 0.98 | 0.87 | 0.92 | 493 |
| Storage | 0.94 | 0.84 | 0.89 | 555 |

**Error analysis**

- **Best performing:** Purchase (F1 0.92), Access (0.89), and Storage (0.89) have distinctive vocabulary that the model picks up easily.
- **Weakest:** Administrative rights has the lowest recall (0.65). It is the smallest class and overlaps in wording with Access, so many of its tickets get misclassified.
- **Hardware** has high recall (0.89) but lower precision (0.78). As the largest class, the model tends to over-predict it.
- **Miscellaneous** is a catch-all category, which naturally makes it harder to separate from the others.

### Example Prediction

```
Ticket: "I cannot access my account and need help logging in."
Predicted Category: Access
```

## ▶️ How to Run

1. Clone the repository
2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Place `all_tickets_processed_improved_v3.csv` in the same folder as the notebook
4. Open the notebook

```bash
jupyter notebook FUTURE_ML_02.ipynb
```

---

## 📌 Future Improvements

- Add **priority tagging** (High / Medium / Low) using keyword rules or a second classifier
- Handle class imbalance (`class_weight="balanced"`, SMOTE) to improve recall for Administrative rights
- Try other models (Linear SVM, Naive Bayes, Random Forest) and compare
- Use n-gram TF-IDF features and hyperparameter tuning (GridSearchCV)
- Try transformer-based models such as BERT for better context understanding
- Deploy as a simple Streamlit app where support agents can paste a ticket and get a category

---

## 👨‍💻 Author

Tushar Thorat

Machine Learning Intern at Future Interns


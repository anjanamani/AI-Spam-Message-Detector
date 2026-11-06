# AI Spam Message Detection

## Project Description
This project uses Machine Learning to detect whether an SMS message is Spam or Ham (Not Spam).

## Technologies Used
- Python
- Machine Learning
- Scikit-learn
- Pandas
- Streamlit

## Project Files
- app.py - Application file
- train_model.py - Model training
- spam_model.pkl - Saved ML model
- requirements.txt - Required libraries
- dataset - SMS dataset

## How to Run
1. Install dependencies:
   pip install -r requirements.txt

2. Run the application:
   streamlit run app.py
   ## Model Performance

Three machine learning algorithms were evaluated:

| Algorithm | Accuracy | Spam F1-score |
|-----------|----------|---------------|
| Naive Bayes | 96.23% | 83.6% |
| Logistic Regression | 96.14% | 83.1% |
| SVM | 98.30% | 93.4% |

Best Performing Model: SVM

- Precision: 97.1%
- Recall: 89.9%
- F1-score: 93.4%
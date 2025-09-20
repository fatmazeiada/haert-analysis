# 🩺 Heart Disease Prediction

## 📌 Overview
This project analyzes patient medical data and predicts the presence of **heart disease** using multiple Machine Learning models.  
The workflow includes **exploratory data analysis (EDA), preprocessing, model training, and evaluation**.

---

## 📊 Dataset
The dataset used is **heart.csv**, which contains medical information such as:
- Age  
- Gender  
- Resting Blood Pressure (`trestbps`)  
- Cholesterol  
- Max Heart Rate Achieved (`thalach`)  
- Target (0 = No Heart Disease, 1 = Heart Disease)  

---

## ⚙️ Workflow
### 🔍 1. Exploratory Data Analysis (EDA)
- Dataset shape, info, and description  
- Missing values check  
- Duplicate detection  
- Visualization:
  - Histograms for distributions  
  - Correlation heatmaps  

### 🛠️ 2. Data Preprocessing
- Encoding categorical values with **LabelEncoder**  
- Combining numerical and categorical features into a single DataFrame  
- Train/Test split (80% / 20%)  

### 🤖 3. Machine Learning Models
The following models were trained and evaluated:
- K-Nearest Neighbors (**KNN**)  
- Decision Tree Classifier  
- Naive Bayes (**GaussianNB**)  
- Support Vector Machine (**SVM**)  
- Random Forest Classifier  
- Gradient Boosting Classifier  

### 📈 4. Model Evaluation
Each model was evaluated using:
- **Confusion Matrix**  
- **Classification Report** (Precision, Recall, F1-score)  

---

## 🛠️ Technologies Used
- **Python 3**  
- **Pandas, NumPy** → Data handling  
- **Matplotlib, Seaborn** → Visualization  
- **Scikit-learn** → Machine Learning  

---

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/YourUsername/heart-disease-prediction.git
   cd heart-disease-prediction

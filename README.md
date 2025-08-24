� Salary Prediction App
 This project is a machine learning-powered Salary Prediction App built using:
 
Python
 XGBoost Regressor (with GridSearchCV tuning)
 Streamlit for the interactive web UI
 One-hot encoding for categorical features
 🚀 Features
 
Interactive Streamlit UI

 User inputs: Age, Experience, Education, Industry, etc.
 Categorical dropdowns with dynamic one-hot encoding
 Real-time salary prediction output
 Custom gradient background using HTML & CSS
 📁 Project Structure
 
 SalaryApp/
 ├── app.py                 # Streamlit UI
 ├── model.pkl              # Trained XGBoost model
 ├── features.pkl           # Feature list used for prediction
 ├── salary_train.csv       # (Optional) Training dataset
 ├── requirements.txt       # Python dependencies
 └── README.md              # Project documentation
 📦 Setup Instructions
 
 1. 
Clone the repository
 git clone https://github.com/akash66-SKY/salary-prediction-app.git
 cd salary-prediction-app
 1. 
Create a virtual environment (optional but recommended)
 1
python-m venv venv
 source venv/bin/activate # or venv\Scripts\activate on Windows
 1. 
Install dependencies
 pip install-r requirements.txt
 1. 
Run the app
 streamlit run app.py
 🧠 Model Training (Optional)
 If you want to retrain the model, use the provided notebook 
 
Cleans data
 Performs one-hot encoding
 Tunes model with GridSearchCV
 Saves the final model and feature list
 Pred_salary.ipynb :
 ✅ Requirements
 streamlit
 pandas
 scikit-learn
 xgboost
 joblib
 📸 Screenshot
 🙌 Credits
 Developed by [Akash Singh]

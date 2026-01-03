[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/blswXyO9)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=20098762&assignment_repo_type=AssignmentRepo)

# An Ensemble Machine Learning Model for Predicting Customer Churn in Retail Banking

## Project Overview
Customer churn poses a significant threat to the profitability and sustainability of retail banks, as acquiring new customers is substantially more costly than retaining existing ones. This project presents an interpretable ensemble machine learning system designed to predict customer churn in retail banking and support data-driven, proactive retention strategies.

The solution combines advanced ensemble learning techniques with SHAP explainability and is deployed through a web-based application that allows stakeholders to upload customer datasets, generate churn predictions, and derive actionable insights.

This project was developed as a Final Year Undergraduate Research Project at Strathmore University, School of Computing and Engineering Sciences.

---

## Objectives
- Predict customer churn likelihood in retail banking using machine learning
- Improve predictive performance using ensemble learning models
- Address class imbalance using SMOTE-based techniques
- Provide model interpretability using SHAP (SHapley Additive exPlanations)
- Deploy the model through a user-friendly web application
- Support collaboration between Retail Banking Admins, Business Analysts, and IT Administrators

---

## Methodology

### Machine Learning Development
The CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology was adopted:
- Business Understanding
- Data Understanding
- Data Preparation
- Modeling
- Evaluation
- Deployment

### Web Application Development
The supporting web-based system was developed using Agile Object-Oriented Analysis and Design (OOAD), allowing for iterative refinement, continuous testing, and modular feature development.

---

## System Architecture
The system consists of five core components:
1. Users (Retail Admins, Business Analysts, IT Admins)
2. Web Application (Dashboard and UI)
3. Backend API (Flask)
4. Machine Learning Model
5. MySQL Database

User requests are handled through the web application and routed via the API to either the database or the machine learning model. Prediction results and SHAP explanations are stored and visualized through role-based dashboards.

---

## Machine Learning Models

### Baseline Models
- Logistic Regression
- Support Vector Machine (SVM)
- Decision Trees

### Advanced Models
- CatBoost
- LightGBM
- XGBoost

### Final Model
- Stacking Ensemble Model

### Performance Highlights
- ROC-AUC: 85.39%
- Recall: 72.97%
- Hyperparameter optimization using Grid Search and Bayesian Optimization

---

## Explainability
To ensure transparency and trust in model predictions, SHAP (SHapley Additive exPlanations) was integrated into the system. SHAP provides feature contribution scores, highlights key churn drivers, and supports customer-level prediction explanations that can be translated into business insights.

---

## Web Application Features
- Secure authentication and role-based access control
- Customer dataset upload (CSV and Excel formats)
- Churn prediction dashboards
- SHAP-based feature importance and explanation views
- Business analyst insight workflow
- System health monitoring for IT administrators

---

## Testing
The system underwent extensive testing, including:
- Functional and non-functional testing
- API endpoint testing
- Access control and security testing
- Model loading and prediction stress testing

All critical test cases passed successfully.

---

## Technologies Used

### Programming and Frameworks
- Python 3.10+
- Flask 2.3.2
- Flask-Session
- Bootstrap

### Machine Learning and Data Science
- scikit-learn
- CatBoost
- XGBoost
- LightGBM
- SHAP
- pandas
- NumPy
- SciPy
- joblib

### Database and Security
- MySQL (XAMPP)
- mysql-connector-python
- bcrypt
- python-dotenv
- email-validator

### Tools
- Git and GitHub
- Google Colab
- Figma

---

## System Requirements

### Minimum Hardware
- RAM: 8 GB (16 GB recommended)
- CPU: Quad-core, 2.0 GHz or higher
- Storage: 128 GB SSD (256 GB recommended)

### Operating System
- Windows 10 or higher

---

## GitHub Branch Structure
- main – Final stable system
- demo – Backup branch for major system changes
- feature/login-signup – Authentication module development
- feature/IT_Admin_Dashboard – IT administration and monitoring features

---

## Recommended Repository Structure
AN-ENSEMBLE-MACHINE-LEARNING-MODEL-FOR-PREDICTING-CUSTOMER-CHURN
│
├── .github/                    # GitHub workflows and configuration
│
├── business_analyst/           # Business analyst dashboards and insight views
├── IT/                         # IT administrator dashboards and system monitoring
├── retail_banking/             # Retail banking admin interfaces
│
├── css/                        # Global and page-specific stylesheets
├── js/                         # Frontend JavaScript logic
│
├── src/                        # Backend application and ML pipeline
│   ├── __pycache__/            # Python cache files
│   ├── analytics_outputs/      # Model outputs, reports, SHAP visualizations
│   ├── app_routes/             # Flask route definitions
│   ├── catboost_info/          # CatBoost training logs and metadata
│   ├── core/                   # Core business logic and utilities
│   ├── flask_session/          # Server-side session storage
│   ├── model_artifacts/        # Trained models and serialized objects
│   ├── uploads/                # Uploaded customer datasets
│   │
│   ├── churn_prediction_model.py   # Model training and inference logic
│   ├── prediction_app.py           # Flask application entry point
│   ├── Churn modelling.csv         # Sample / training dataset
│   ├── requirements.txt            # Python dependencies
│   ├── .env.example                # Environment variable template
│   └── .env                        # Environment variables (ignored in Git)
│
├── login.html                  # Authentication page
├── user_verification.html      # User verification interface
├── login_background.svg        # UI asset
│
├── .gitignore
└── README.md


---

## Future Work
- Integration with live banking systems
- Automated model retraining pipelines
- Exploration of deep learning architectures
- Real-time streaming data support
- Expanded explainability dashboards

---

## Author
Manya George Okech  
BSc. Informatics and Computer Science  
Strathmore University, Nairobi, Kenya

---

## Supervisor
Mr. Deperias Kerre

---

## License
This project is intended for academic and research purposes.

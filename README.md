# 🌸 MLflow MLOps Project — Iris Classification

## 📘 Overview
This project demonstrates the **end-to-end Machine Learning workflow** using **MLflow**, from model training to model tracking and versioning.  
We use the **Iris dataset** for multi-class classification and compare multiple algorithms including Logistic Regression, SVM, Decision Tree, Random Forest, and AdaBoost.

## 🚀 Project Objectives
- Train multiple machine learning models on the Iris dataset  
- Log parameters, metrics, and models using **MLflow Tracking**  
- Compare model performance through the MLflow UI  
- Register the best model in **MLflow Model Registry**  
- Load the registered model and make predictions  
- Manage source code via **Git + GitHub (DagsHub ready)**  

## 🧠 Machine Learning Models Used
| Model | Key Parameters |
|--------|----------------|
| Logistic Regression | Default |
| SVC | `C=0.0001`, `kernel='rbf'` |
| Random Forest | `max_depth=5`, `max_leaf_nodes=2`, `max_features=2` |
| Decision Tree | `max_depth=3`, `min_samples_leaf=2`, `min_samples_split=4` |
| AdaBoost | Base estimator = `DecisionTree(max_depth=2)`, `n_estimators=40`, `learning_rate=0.01` |

## 🧩 Project Structure
```
ML-project-1/
│
├── main.py / notebook.ipynb        # Main ML training script
├── requirements.txt                 # Dependencies
├── README.md                        # Project documentation
├── mlruns/                          # MLflow tracking data
└── models/                          # Saved or registered models
```

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/PrajwalDataScientist/MLDagsHub.git
cd MLDagsHub
```

### 2️⃣ Create & Activate Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate    # For Windows
# source venv/bin/activate   # For Linux/Mac
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run MLflow Tracking Server
```bash
mlflow ui
```
By default, MLflow UI runs at:  
👉 **http://127.0.0.1:5000**

## 🧮 Model Training and Logging
Each model is trained and evaluated using the following metrics:
- **Train Accuracy**
- **Test Accuracy**

Metrics and models are logged to MLflow as:
```python
mlflow.log_param("model_name", name)
mlflow.log_metric("accuracy", model['train_accuracy'])
mlflow.sklearn.log_model(mode_list[name], artifact_path="model")
```

## 🏷️ Model Registry
The best model (AdaBoost) is registered for versioning:
```python
model_uri = "runs:/<run_id>/model"
result = mlflow.register_model(model_uri, "AdaBoostClassifier")
```

Later, it can be reloaded as:
```python
from mlflow import sklearn
model_loaded = sklearn.load_model("models:/AdaBoostClassifier/4")
y_pred = model_loaded.predict(X_test)
```

## 📊 Results Summary
| Model | Train Accuracy | Test Accuracy |
|--------|----------------|---------------|
| Logistic Regression | ~97% | ~100% |
| SVC | ~95% | ~98% |
| Decision Tree | ~96% | ~99% |
| Random Forest | ~99% | ~100% |
| AdaBoost | ~96% | ~100% |

✅ **AdaBoostClassifier** was selected as the best performing model.

## 🌐 GitHub + DagsHub Integration (Optional)
You can connect this repo to **DagsHub** for remote MLflow tracking:
```bash
mlflow.set_tracking_uri("https://dagshub.com/<username>/<repo>.mlflow")
```

This allows automatic experiment logging and visualization directly on DagsHub.

## 🏁 Conclusion
This project successfully showcases:
- A complete **MLOps pipeline** for experimentation tracking  
- Automated model logging and versioning via **MLflow**  
- Reproducibility and transparency in model management  
- Git-based version control and CI/CD-ready structure

## ✨ Author
**👨‍💻 Prajwal B**  
Machine Learning & Data Science Enthusiast  
  

## 🧰 Requirements
```
pandas
scikit-learn
mlflow
```


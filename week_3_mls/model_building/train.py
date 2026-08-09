# for data manipulation
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
# for model serialization
import joblib
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlops-training-experiment")

# --- Load dataset ---
df = pd.read_csv("week_3_mls/data/tourism.csv")
print("Dataset loaded successfully.")

# Drop ID column
df = df.drop(columns=["CustomerID"])

# Target variable
target_col = "ProdTaken"
X = df.drop(columns=[target_col])
y = df[target_col]

# Train/test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature groups
numeric_features = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips",
    "Passport", "PitchSatisfactionScore", "OwnCar",
    "NumberOfChildrenVisiting", "MonthlyIncome"
]
categorical_features = [
    "TypeofContact", "Occupation", "Gender", "ProductPitched",
    "MaritalStatus", "Designation"
]

# Handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# Preprocessing
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features)
)

# Base XGBoost model
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

# Hyperparameter grid
param_grid = {
    "xgbclassifier__n_estimators": [50, 75, 100],
    "xgbclassifier__max_depth": [2, 3, 4],
    "xgbclassifier__learning_rate": [0.01, 0.05, 0.1],
    "xgbclassifier__colsample_bytree": [0.5, 0.6],
    "xgbclassifier__reg_lambda": [0.5, 0.6],
}

# Pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# --- Training & Logging ---
with mlflow.start_run():
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Log all parameter combinations
    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        param_set = results["params"][i]
        mean_score = results["mean_test_score"][i]
        std_score = results["std_test_score"][i]

        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)

    # Log best params
    mlflow.log_params(grid_search.best_params_)

    # Best model
    best_model = grid_search.best_estimator_
    print("Best params:", grid_search.best_params_)

    classification_threshold = 0.45

    y_pred_train = (best_model.predict_proba(Xtrain)[:, 1] >= classification_threshold).astype(int)
    y_pred_test = (best_model.predict_proba(Xtest)[:, 1] >= classification_threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)
    print(classification_report(ytest, y_pred_test))

    mlflow.log_metrics({
        "train_accuracy": train_report["accuracy"],
        "train_precision": train_report["1"]["precision"],
        "train_recall": train_report["1"]["recall"],
        "train_f1-score": train_report["1"]["f1-score"],
        "test_accuracy": test_report["accuracy"],
        "test_precision": test_report["1"]["precision"],
        "test_recall": test_report["1"]["recall"],
        "test_f1-score": test_report["1"]["f1-score"]
    })

    # Save model for Streamlit app
    os.makedirs("week_3_mls/deployment", exist_ok=True)
    model_path = "week_3_mls/deployment/best_tourism_model_v1.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved to {model_path}")

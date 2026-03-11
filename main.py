import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.preprocessing import load_data, handle_missing_values, remove_duplicates, remove_outliers_iqr
from src.feature_engineering import create_features
from src.models import get_models
from src.train_models import train_and_evaluate
from src.experiment_tracker import save_results, select_best_model


# Load dataset
data = load_data("data/tourism_dataset.csv")


# Data preprocessing
data = handle_missing_values(data)
data = remove_duplicates(data)


# Feature engineering
data = create_features(data)


# Define target variable
target = "Inbound-Total Arrival(overnight stay)( In thousands)"


# Define feature columns
features = data.select_dtypes(include=["float64", "int64"]).columns.tolist()
features.remove(target)


X = data[features]
y = data[target]


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Feature scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Load models
models = get_models()


results = []


# Train and evaluate models
for model_name, model in models.items():

    rmse, mae, r2 = train_and_evaluate(
        model, X_train, y_train, X_test, y_test
    )

    results.append({
        "Model": model_name,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    })


# Save experiment results
results_df = save_results(results)


# Select best model
best_model = select_best_model(results_df)


print("Best Model:", best_model)
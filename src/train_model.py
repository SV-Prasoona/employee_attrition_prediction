import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from src.data_preprocessing import load_data, clean_data
from src.feature_engineering import encode_features, split_features, scale_features
from src.evaluate_model import evaluate_model
from src.utils import save_object


def train():

    print("Loading dataset...")
    df = load_data("data/raw/Employee_Attrition_DataSet.csv")

    print("Cleaning dataset...")
    df = clean_data(df)

    print("Encoding categorical features...")
    df = encode_features(df)

    print("Splitting features and target...")
    X, y = split_features(df)

    print("Scaling features...")
    X, scaler = scale_features(X)

    print("Splitting train-test data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training Random Forest Model...")
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test)

    evaluate_model(y_test, y_pred)

    print("Saving model and scaler...")
    os.makedirs("models", exist_ok=True)

    save_object(model, "models/model.pkl")
    save_object(scaler, "models/scaler.pkl")

    print("✅ Training complete and model saved!")
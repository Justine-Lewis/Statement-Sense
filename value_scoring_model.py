"""
Scoring Model using Random Forest Classifier

"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier


def load_data(filepath: str) -> pd.DataFrame:
    """Load the simulated subscription dataset."""
    return pd.read_csv(filepath)


def prepare_features(df: pd.DataFrame):
    """Select model features and target label."""
    feature_columns = [
        "n_charges",
        "avg_monthly_usd",
        "usage_intensity",
        "trial_prob",
        "duration_months",
        "duration_score",
        "cost_efficiency",
        "user_priority"
    ]

    X = df[feature_columns]
    y = df["risk_label"]

    return X, y, feature_columns


def train_model(X_train, y_train) -> RandomForestClassifier:
    """Train a balanced Random Forest classifier."""
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Print evaluation metrics."""
    y_pred = model.predict(X_test)

    print("\nModel Accuracy:")
    print(f"{accuracy_score(y_test, y_pred):.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return y_pred


def show_feature_importance(model, feature_names):
    """Display ranked feature importances."""
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    print("\nFeature Importances:")
    print(importance_df.to_string(index=False))

    return importance_df


def show_sample_predictions(X_test, y_test, y_pred, n=10):
    """Show a few sample predictions against actual labels."""
    sample_df = X_test.head(n).copy()
    sample_df["actual_risk"] = y_test.head(n).values
    sample_df["predicted_risk"] = y_pred[:n]

    print("\nSample Predictions:")
    print(sample_df.to_string(index=False))


if __name__ == "__main__":
    # Load data
    df = load_data("simulated_subscriptions.csv")

    # Prepare features and target
    X, y, feature_names = prepare_features(df)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    y_pred = evaluate_model(model, X_test, y_test)

    # Show feature importance
    show_feature_importance(model, feature_names)

    # Show sample predictions
    show_sample_predictions(X_test, y_test, y_pred, n=10)

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Test samples:     {len(X_test)}")
    print(f"Classes: {sorted(y.unique())}")
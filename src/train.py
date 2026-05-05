import pandas as pd
import mlflow
import mlflow.sklearn
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from mlflow.models.signature import infer_signature

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


def load_and_preprocess(path):
    df = pd.read_csv(path)

    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df.drop(columns=["Cabin", "Ticket", "Name", "PassengerId"], inplace=True)

    le = LabelEncoder()
    df["Sex"] = le.fit_transform(df["Sex"])
    df["Embarked"] = le.fit_transform(df["Embarked"])

    X = df.drop("Survived", axis=1)
    y = df["Survived"]
    return X, y


def train():
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    X, y = load_and_preprocess(config["data"]["path"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"]
    )

    params = config["model"]

    with mlflow.start_run():
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)

        signature = infer_signature(X_train, model.predict(X_train))
        input_example = X_train.head(3)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=config["artifacts"]["model_path"],
            signature=signature,
            input_example=input_example
        )

        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC:  {auc:.4f}")


if __name__ == "__main__":
    train()
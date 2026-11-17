import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():

    df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'dataset.csv'))

    features = ['score_z1', 'res_length', 'res_words', 'vac_words', 'length_diff', 'has_experience', 'has_education']
    X = df[features]
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    print(f"Обучающая часть: {len(X_train)}")
    print(f"Проверочная часть: {len(X_test)}")
    print(f"Доля положительных в тесте: {y_test.sum()} из {len(y_test)}")

    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # Вероятности

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\n--- МЕТРИКИ ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    print("\nМатрица ошибок (Confusion Matrix):")
    print(cm)
    print(f"Верных отказов (TN): {cm[0][0]}")
    print(f"Ложных срабатываний (FP): {cm[0][1]}")
    print(f"Пропусков (FN): {cm[1][0]}")
    print(f"Верных срабатываний (TP): {cm[1][1]}")


if __name__ == "__main__":
    main()
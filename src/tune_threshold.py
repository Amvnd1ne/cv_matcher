import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'dataset.csv'))
    features = ['score_z1', 'res_length', 'res_words', 'vac_words', 'length_diff', 'has_experience', 'has_education']
    X = df[features]
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    model.fit(X_train, y_train)

    # Получаем вероятности на проверочной части
    y_proba = model.predict_proba(X_test)[:, 1]

    print("Порог | Precision | Recall | F1 | Кол-во предсказанных 1")
    print("-" * 60)

    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        count_positive = sum(y_pred)
        print(f"{thresh:.1f}   | {prec:.2f}      | {rec:.2f}   | {f1:.2f} | {count_positive}")


if __name__ == "__main__":
    main()
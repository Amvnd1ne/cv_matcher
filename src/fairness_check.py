import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import read_resume, read_vacancy
from train_ranker import extract_features

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESUMES_DIR = os.path.join(PROJECT_ROOT, 'data', 'resumes')
VACANCIES_DIR = os.path.join(PROJECT_ROOT, 'data', 'vacancies')


def get_group(resume_text):
    """Определяет группу по признакам из резюме (не защищённым)."""
    words_count = len(resume_text.split())
    has_exp = "опыт" in resume_text.lower()

    # Используем другой порог: 30 слов (наши резюме короткие)
    if words_count < 30:
        return "Очень короткое резюме (<30 слов)"
    elif not has_exp:
        return "Без слова 'опыт'"
    else:
        return "Резюме с опытом (30+ слов)"


def main():
    # 1. Загружаем разметку
    df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'ranking_dataset.csv'))

    # 2. Извлекаем признаки и группы
    features = []
    labels = []
    groups = []
    resume_names = []

    for _, row in df.iterrows():
        res_text = read_resume(os.path.join(RESUMES_DIR, row['resume']))
        vac_data = read_vacancy(os.path.join(VACANCIES_DIR, row['vacancy']))
        if not res_text or not vac_data:
            continue

        vac_text = vac_data['требования'] + " " + vac_data['обязанности']
        feats = extract_features(res_text, vac_text)

        features.append(feats)
        labels.append(row['grade'])
        groups.append(get_group(res_text))
        resume_names.append(row['resume'])

    X = np.array(features)
    y = np.array(labels)
    groups = np.array(groups)

    # 3. Обучаем модель на всех данных (для демонстрации)
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    # 4. Предсказываем
    y_pred_proba = model.predict_proba(X)[:, 2]

    # 5. Считаем среднюю позицию и долю подходящих в топ-5 для каждой группы
    print("Разрезы по группам (fairness):")
    print(f"{'Группа':<35} {'Пар':<6} {'Ср. позиция':<12} {'Доля 2 в топ-5':<15}")
    print("-" * 75)

    for group in sorted(set(groups)):
        mask = groups == group
        group_indices = np.where(mask)[0]

        # Средняя позиция (ранг) внутри группы
        # Для простоты считаем средний балл
        avg_score = y_pred_proba[group_indices].mean()

        # Доля подходящих (класс 2) в группе
        share_class_2 = (y[group_indices] == 2).mean()

        print(f"{group:<35} {mask.sum():<6} {avg_score:<12.3f} {share_class_2:<15.2f}")


if __name__ == "__main__":
    main()
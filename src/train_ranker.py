import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ndcg_score
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import read_resume, read_vacancy
from vectorizer import calculate_vector_score
from text_processor import normalize_text

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESUMES_DIR = os.path.join(PROJECT_ROOT, 'data', 'resumes')
VACANCIES_DIR = os.path.join(PROJECT_ROOT, 'data', 'vacancies')


def extract_features(resume_text, vacancy_text):
    """Извлекает признаки для ранжировщика."""
    score_data = calculate_vector_score(resume_text, vacancy_text)
    score = score_data['score']

    key_skills = {"sql", "python", "excel", "pandas", "numpy", "bi", "tableau", "power"}
    res_words = set(normalize_text(resume_text))
    vac_words = set(normalize_text(vacancy_text))
    matched_skills = len(key_skills.intersection(res_words).intersection(vac_words))

    res_length = len(resume_text.split())
    has_experience = 1 if "опыт" in resume_text.lower() else 0
    has_education = 1 if "образование" in resume_text.lower() else 0

    return [score, matched_skills, res_length, has_experience, has_education]


def check_calibration(y_true, y_proba, bins=10):
    """Проверяет калибровку модели."""
    bin_edges = np.linspace(0, 1, bins + 1)
    print("\nКалибровка (группы оценок):")
    print(f"{'Группа':<15} {'Пар':<8} {'Наблюдаемая доля 2':<20}")
    print("-" * 45)

    for i in range(bins):
        mask = (y_proba >= bin_edges[i]) & (y_proba < bin_edges[i + 1])
        if mask.sum() > 0:
            observed = (y_true[mask] == 2).mean()
            print(f"{bin_edges[i]:.1f}–{bin_edges[i+1]:.1f}     {mask.sum():<8} {observed:.2f}")


def main():
    df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'ranking_dataset.csv'))

    features = []
    labels = []

    for _, row in df.iterrows():
        res_text = read_resume(os.path.join(RESUMES_DIR, row['resume']))
        vac_data = read_vacancy(os.path.join(VACANCIES_DIR, row['vacancy']))
        if not res_text or not vac_data:
            continue

        vac_text = vac_data['требования'] + " " + vac_data['обязанности']
        feats = extract_features(res_text, vac_text)

        features.append(feats)
        labels.append(row['grade'])

    X = np.array(features)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 2]

    ndcg = ndcg_score([y_test], [y_pred_proba], k=5)
    print(f"NDCG@5 на тесте: {ndcg:.3f}")

    print("\nВажность признаков:")
    feature_names = ['score_z1', 'matched_skills', 'res_length', 'has_experience', 'has_education']
    for name, coef in zip(feature_names, model.coef_[2]):
        print(f"  {name}: {coef:.3f}")

    # Вызываем калибровку (функция уже определена выше)
    check_calibration(y_test, y_pred_proba)


if __name__ == "__main__":
    main()
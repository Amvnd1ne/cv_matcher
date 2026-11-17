import os
import pandas as pd
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import read_resume, read_vacancy
from vectorizer import calculate_vector_score
from text_processor import normalize_text

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESUMES_DIR = os.path.join(PROJECT_ROOT, 'data', 'resumes')
VACANCIES_DIR = os.path.join(PROJECT_ROOT, 'data', 'vacancies')


def rule_for_target(resume_text, vacancy_text):
    """
    Правило разметки:
    1 - если в резюме есть хотя бы 2 ключевых навыка из вакансии (sql, python, excel, аналитика)
    0 - иначе.
    """
    res_words = set(normalize_text(resume_text))
    vac_words = set(normalize_text(vacancy_text))

    key_skills = {"sql", "python", "excel", "аналитика", "данные", "pandas", "numpy"}

    matched_skills = key_skills.intersection(res_words).intersection(vac_words)

    if len(matched_skills) >= 2:
        return 1
    return 0


def main():
    resumes = {}
    for f in os.listdir(RESUMES_DIR):
        if f.endswith('.txt'):
            text = read_resume(os.path.join(RESUMES_DIR, f))
            if text:
                resumes[f] = text

    vacancies = {}
    for f in os.listdir(VACANCIES_DIR):
        if f.endswith('.json'):
            data = read_vacancy(os.path.join(VACANCIES_DIR, f))
            if data:
                vacancies[f] = data

    print(f"Резюме: {len(resumes)}, Вакансий: {len(vacancies)}")

    rows = []
    for res_name, res_text in resumes.items():
        for vac_name, vac_data in vacancies.items():
            vac_text = vac_data['требования'] + " " + vac_data['обязанности']

            score_data = calculate_vector_score(res_text, vac_text)
            score = score_data['score']

            res_length = len(res_text)
            res_words_count = len(res_text.split())
            vac_words_count = len(vac_text.split())
            length_diff = res_length - len(vac_text)

            has_experience = 1 if "опыт" in res_text.lower() else 0
            has_education = 1 if "образование" in res_text.lower() else 0

            target = rule_for_target(res_text, vac_text)

            rows.append({
                'resume': res_name,
                'vacancy': vac_name,
                'score_z1': score,
                'res_length': res_length,
                'res_words': res_words_count,
                'vac_words': vac_words_count,
                'length_diff': length_diff,
                'has_experience': has_experience,
                'has_education': has_education,
                'target': target
            })

    df = pd.DataFrame(rows)

    print("\nРаспределение целевой переменной:")
    print(df['target'].value_counts())

    output_path = os.path.join(PROJECT_ROOT, 'data', 'dataset.csv')
    df.to_csv(output_path, index=False)

    print(f"\nТаблица сохранена: {output_path}")
    print(f"Размер таблицы: {df.shape}")


if __name__ == "__main__":
    main()
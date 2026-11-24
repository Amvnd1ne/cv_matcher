import os
import pandas as pd
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import read_resume, read_vacancy
from text_processor import normalize_text

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESUMES_DIR = os.path.join(PROJECT_ROOT, 'data', 'resumes')
VACANCIES_DIR = os.path.join(PROJECT_ROOT, 'data', 'vacancies')


def get_relevance_grade(resume_text, vacancy_data):
    """
    Правило разметки (3 градации):
    2 - Полное соответствие (есть все ключевые навыки)
    1 - Частичное (есть часть навыков, но уровень не тот)
    0 - Не подходит (нет ключевых навыков)
    """
    res_words = set(normalize_text(resume_text))
    vac_text = vacancy_data['требования'] + " " + vacancy_data['обязанности']
    vac_words = set(normalize_text(vac_text))

    # Ключевые навыки для нашей области (аналитика)
    key_skills = {"sql", "python", "excel", "pandas", "numpy", "bi", "tableau", "power"}

    # Сколько ключевых навыков из вакансии есть в резюме
    matched = key_skills.intersection(res_words).intersection(vac_words)

    # Если есть 2 и более ключевых навыка -> "Подходит" (2)
    if len(matched) >= 2:
        # Проверяем уровень (если в вакансии "junior", а в резюме "senior" - это 1)
        if "junior" in vac_text.lower() and ("senior" in resume_text.lower() or "3 года" in resume_text.lower()):
            return 1
        return 2

    # Если есть 1 навык -> "Частично" (1)
    elif len(matched) == 1:
        return 1

    # Если нет навыков -> "Не подходит" (0)
    return 0


def main():
    # Загружаем данные
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

    # Создаем таблицу
    rows = []
    for res_name, res_text in resumes.items():
        for vac_name, vac_data in vacancies.items():
            grade = get_relevance_grade(res_text, vac_data)
            rows.append({
                'resume': res_name,
                'vacancy': vac_name,
                'grade': grade
            })

    df = pd.DataFrame(rows)

    # Выводим статистику
    print("Распределение градаций:")
    print(df['grade'].value_counts())
    print(f"\nВсего пар: {len(df)}")

    # Сохраняем
    output_path = os.path.join(PROJECT_ROOT, 'data', 'ranking_dataset.csv')
    df.to_csv(output_path, index=False)
    print(f"Таблица сохранена: {output_path}")


if __name__ == "__main__":
    main()
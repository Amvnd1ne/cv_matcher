import os
import sys
import random
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import read_resume, read_vacancy
from vectorizer import calculate_vector_score
from matcher import calculate_overlap
from llm_agent import get_recommendations

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESUMES_DIR = os.path.join(PROJECT_ROOT, 'data', 'resumes')
VACANCIES_DIR = os.path.join(PROJECT_ROOT, 'data', 'vacancies')


def main():
    # Собираем все резюме и вакансии
    resumes = []
    for f in os.listdir(RESUMES_DIR):
        if f.endswith('.txt'):
            text = read_resume(os.path.join(RESUMES_DIR, f))
            if text:
                resumes.append((f, text))

    vacancies = []
    for f in os.listdir(VACANCIES_DIR):
        if f.endswith('.json'):
            data = read_vacancy(os.path.join(VACANCIES_DIR, f))
            if data:
                vacancies.append((f, data))

    print(f"Доступно резюме: {len(resumes)}, вакансий: {len(vacancies)}")
    print(f"Всего возможных пар: {len(resumes) * len(vacancies)}")
    print("-" * 60)

    # Журнал прогонов
    log = []
    failures = 0

    for i in range(1, 51):
        # Выбираем случайную пару
        res_name, res_text = random.choice(resumes)
        vac_name, vac_data = random.choice(vacancies)
        vac_text = vac_data['требования'] + " " + vac_data['обязанности']

        try:
            # Считаем балл
            score_data = calculate_vector_score(res_text, vac_text)
            score = score_data['score']

            # Находим недостающие слова
            overlap = calculate_overlap(res_text, vac_text)
            missing_words = overlap['missing_words'][:10]

            # Получаем рекомендации
            result = get_recommendations(res_text, vac_text, score, overlap['matched_words'], missing_words)

            # Проверяем, что ответ валиден
            if not result.get('analysis') or not result.get('advice'):
                raise ValueError("Пустой ответ")

            log.append({
                'run': i,
                'resume': res_name,
                'vacancy': vac_name,
                'score': round(score, 2),
                'status': 'OK',
                'advice_count': len(result['advice'])
            })
            print(f"Прогон {i:2d}/50: ✅ OK (балл {score:.2f}, советов {len(result['advice'])})")

        except Exception as e:
            failures += 1
            log.append({
                'run': i,
                'resume': res_name,
                'vacancy': vac_name,
                'score': 0,
                'status': f'ERROR: {e}',
                'advice_count': 0
            })
            print(f"Прогон {i:2d}/50: ❌ ОШИБКА: {e}")

    print("-" * 60)
    print(f"Успешных прогонов: {50 - failures}/50")
    print(f"Ошибок: {failures}/50")

    # Сохраняем журнал в файл
    log_path = os.path.join(PROJECT_ROOT, 'run_log.md')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("# Журнал прогонов AI-агента (50 запусков)\n\n")
        f.write("| № | Резюме | Вакансия | Балл | Статус | Советов |\n")
        f.write("|---|---|---|---|---|---|\n")
        for entry in log:
            f.write(
                f"| {entry['run']} | {entry['resume']} | {entry['vacancy']} | {entry['score']} | {entry['status']} | {entry['advice_count']} |\n")

    print(f"\nЖурнал сохранен: {log_path}")


if __name__ == "__main__":
    main()
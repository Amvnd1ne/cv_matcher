import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import read_resume, read_vacancy
from vectorizer import calculate_vector_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESUMES_DIR = os.path.join(PROJECT_ROOT, 'data', 'resumes')
VACANCIES_DIR = os.path.join(PROJECT_ROOT, 'data', 'vacancies')


def main():

    resumes = {}
    for filename in os.listdir(RESUMES_DIR):
        if filename.endswith('.txt'):
            path = os.path.join(RESUMES_DIR, filename)
            text = read_resume(path)
            if text:
                resumes[filename] = text

    vacancies = {}
    for filename in os.listdir(VACANCIES_DIR):
        if filename.endswith('.json'):
            path = os.path.join(VACANCIES_DIR, filename)
            data = read_vacancy(path)
            if data:
                vacancies[filename] = data

    print(f"Найдено резюме: {len(resumes)}")
    print(f"Найдено вакансий: {len(vacancies)}")
    print("-" * 80)

    results = []
    for res_name, res_text in resumes.items():
        for vac_name, vac_data in vacancies.items():
            vac_text = vac_data['требования'] + " " + vac_data['обязанности']
            score_data = calculate_vector_score(res_text, vac_text)
            results.append({
                'resume': res_name,
                'vacancy': vac_name,
                'vacancy_title': vac_data['должность'],
                'score': score_data['score']
            })

    print(f"{'Резюме':<25} {'Вакансия':<40} {'Балл':>6}")
    print("-" * 80)
    for r in results:
        print(f"{r['resume']:<25} {r['vacancy_title'][:38]:<40} {r['score']:>6.3f}")

    print("-" * 80)
    print(f"Всего пар: {len(results)}")
    print(f"Средний балл: {sum(r['score'] for r in results) / len(results):.4f}")
    print(f"Минимальный балл: {min(r['score'] for r in results):.4f}")
    print(f"Максимальный балл: {max(r['score'] for r in results):.4f}")


if __name__ == "__main__":
    main()
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import read_resume, read_vacancy
from vectorizer import calculate_vector_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESUMES_DIR = os.path.join(PROJECT_ROOT, 'data', 'resumes')
VACANCIES_DIR = os.path.join(PROJECT_ROOT, 'data', 'vacancies')


def main():

    resumes = []
    for f in os.listdir(RESUMES_DIR):
        if f.endswith('.txt'):
            text = read_resume(os.path.join(RESUMES_DIR, f))
            if text:
                resumes.append(text)


    vacancies = []
    for f in os.listdir(VACANCIES_DIR):
        if f.endswith('.json'):
            data = read_vacancy(os.path.join(VACANCIES_DIR, f))
            if data:
                vacancy_text = data['требования'] + " " + data['обязанности']
                vacancies.append(vacancy_text)

    total_pairs = len(resumes) * len(vacancies)
    print(f"Резюме: {len(resumes)}, Вакансий: {len(vacancies)}")
    print(f"Всего пар: {total_pairs}")


    start = time.time()
    for res in resumes:
        for vac in vacancies:
            calculate_vector_score(res, vac)
    elapsed = time.time() - start

    print(f"\nВремя полного прогона ({total_pairs} пар): {elapsed:.4f} сек")
    print(f"Время на 100 пар (оценка): {elapsed / total_pairs * 100:.4f} сек")
    print(f"Время на 1000 пар (оценка): {elapsed / total_pairs * 1000:.4f} сек")


if __name__ == "__main__":
    main()
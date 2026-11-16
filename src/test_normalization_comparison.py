import os
import sys
import re
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import read_resume, read_vacancy

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESUMES_DIR = os.path.join(PROJECT_ROOT, 'data', 'resumes')
VACANCIES_DIR = os.path.join(PROJECT_ROOT, 'data', 'vacancies')

STOP_WORDS = {"и", "в", "на", "с", "по", "для", "от", "до", "из", "к", "у", "о", "об", "за", "под", "над", "при", "без",
              "через", "между", "или", "но", "а", "что", "чтобы", "как", "так", "это", "весь", "все", "всех", "свой",
              "свои", "который", "которая", "которые", "которое", "быть", "был", "была", "было", "были", "есть", "не",
              "ни", "же", "ли", "бы"}


def normalize_v1(text):
    return text.lower().split()


def normalize_v2(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    words = text.split()
    return [w for w in words if w not in STOP_WORDS]


def normalize_v3(text):

    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in STOP_WORDS]

    return [re.sub(r'(а|ы|и|я|ов|ев|ый|ий|ая|ое|ые|ых)$', '', w) for w in words]


def calculate_score(resume_words, vacancy_words):
    resume_set = set(resume_words)
    vacancy_set = set(vacancy_words)
    if len(vacancy_set) == 0:
        return 0.0
    return len(resume_set.intersection(vacancy_set)) / len(vacancy_set)


def main():

    resumes = []
    for f in os.listdir(RESUMES_DIR):
        if f.endswith('.txt'):
            text = read_resume(os.path.join(RESUMES_DIR, f))
            if text: resumes.append(text)

    vacancies = []
    for f in os.listdir(VACANCIES_DIR):
        if f.endswith('.json'):
            data = read_vacancy(os.path.join(VACANCIES_DIR, f))
            if data: vacancies.append(data['требования'] + " " + data['обязанности'])


    for name, func in [("v1 (только регистр)", normalize_v1),
                       ("v2 (стоп-слова)", normalize_v2),
                       ("v3 (лемматизация)", normalize_v3)]:
        start = time.time()
        scores = []
        for res in resumes:
            for vac in vacancies:
                r_words = func(res)
                v_words = func(vac)
                scores.append(calculate_score(r_words, v_words))
        elapsed = time.time() - start

        print(f"\n--- {name} ---")
        print(f"Время: {elapsed:.4f} сек")
        print(f"Средний балл: {sum(scores) / len(scores):.4f}")
        print(f"Строк кода: {len(func.__code__.co_code) // 2}")  # грубая оценка


if __name__ == "__main__":
    main()
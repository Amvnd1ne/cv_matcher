import os
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessor import preprocess_text
from data_loader import read_resume, read_vacancy

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_DIR = os.path.join(PROJECT_ROOT, 'data', 'corpus')
RESUMES_DIR = os.path.join(PROJECT_ROOT, 'data', 'resumes')


def main():

    vacancies = []
    for f in os.listdir(CORPUS_DIR):
        if f.endswith('.json'):
            with open(os.path.join(CORPUS_DIR, f), 'r', encoding='utf-8') as file:
                data = json.load(file)
                vacancies.append(data['требования'] + " " + data['обязанности'])

    resumes = []
    for f in os.listdir(RESUMES_DIR):
        if f.endswith('.txt'):
            text = read_resume(os.path.join(RESUMES_DIR, f))
            if text:
                resumes.append(text)

    processed_vacancies = [preprocess_text(v) for v in vacancies]
    processed_resumes = [preprocess_text(r) for r in resumes]

    vectorizer = TfidfVectorizer(min_df=2)
    X_vac = vectorizer.fit_transform(processed_vacancies)
    X_res = vectorizer.transform(processed_resumes)

    print("--- тест 1 - Документ сам с собой ---")

    sim_self = cosine_similarity(X_vac[0], X_vac[0])[0][0]
    print(f"Ожидалось: 1.0, Получилось: {sim_self:.4f}")

    print("\n--- тест 2 - Два документа без общих слов ---")

    fake_doc = preprocess_text("зеленый слоник прыгает на луне")
    X_fake = vectorizer.transform([fake_doc])
    sim_fake = cosine_similarity(X_vac[0], X_fake)[0][0]
    print(f"Ожидалось: 0.0, Получилось: {sim_fake:.4f}")

    print("\n--- ТЕСТ 3: Реальная пара (Резюме и Вакансия) ---")

    sim_real = cosine_similarity(X_res[0], X_vac[0])[0][0]
    print(f"Близость между резюме и вакансией: {sim_real:.4f}")


if __name__ == "__main__":
    main()
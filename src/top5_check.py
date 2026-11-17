import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import read_resume, read_vacancy

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESUMES_DIR = os.path.join(PROJECT_ROOT, 'data', 'resumes')
VACANCIES_DIR = os.path.join(PROJECT_ROOT, 'data', 'vacancies')


def main():
    print("Загрузка модели...")
    model = SentenceTransformer('intfloat/multilingual-e5-small')

    # Загружаем 15 реальных вакансий
    vacancies = {}
    for f in os.listdir(VACANCIES_DIR):
        if f.endswith('.json'):
            data = read_vacancy(os.path.join(VACANCIES_DIR, f))
            if data:
                vacancies[f] = data['требования'] + " " + data['обязанности']

    vac_embeddings = {}
    for f, text in vacancies.items():
        vac_embeddings[f] = model.encode("passage: " + text)

    # Выбираем 3 резюме для проверки
    check_resumes = ['resume_strong.txt', 'resume_related.txt', 'resume_weak.txt']

    for res_name in check_resumes:
        res_text = read_resume(os.path.join(RESUMES_DIR, res_name))
        if not res_text:
            continue

        res_emb = model.encode("query: " + res_text)

        scores = []
        for vac_name, vac_emb in vac_embeddings.items():
            sim = cosine_similarity(res_emb.reshape(1, -1), vac_emb.reshape(1, -1))[0][0]
            scores.append((sim, vac_name, vacancies[vac_name]))

        # Сортируем по убыванию близости
        scores.sort(key=lambda x: x[0], reverse=True)

        print(f"\n=== Резюме: {res_name} ===")
        print(f"{'Близость':<10} | {'Вакансия':<40} | {'Подходит?'}")
        print("-" * 70)

        for sim, vac_name, vac_text in scores[:5]:
            # Здесь вы можете вручную оценить, подходит ли вакансия
            print(f"{sim:.3f}      | {vac_name:<40} | ?")


if __name__ == "__main__":
    main()
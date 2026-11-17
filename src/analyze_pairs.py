import os
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessor import preprocess_text
from data_loader import read_resume

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_DIR = os.path.join(PROJECT_ROOT, 'data', 'corpus')
RESUMES_DIR = os.path.join(PROJECT_ROOT, 'data', 'resumes')


def main():
    # Загружаем данные
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

    # Предобработка
    processed_vacancies = [preprocess_text(v) for v in vacancies]
    processed_resumes = [preprocess_text(r) for r in resumes]

    # Векторизация TF-IDF
    vectorizer = TfidfVectorizer(min_df=2)
    X_vac = vectorizer.fit_transform(processed_vacancies)
    X_res = vectorizer.transform(processed_resumes)

    feature_names = vectorizer.get_feature_names_out()

    # Ищем самую высокую и самую низкую близость
    max_sim = -1
    min_sim = 2
    max_pair = None
    min_pair = None

    for i, res in enumerate(processed_resumes):
        for j, vac in enumerate(processed_vacancies[:50]):  # Ограничимся 50 для скорости
            sim = cosine_similarity(X_res[i], X_vac[j])[0][0]
            if sim > max_sim:
                max_sim = sim
                max_pair = (i, j)
            if sim < min_sim:
                min_sim = sim
                min_pair = (i, j)

    # Функция для вывода топ-5 слов
    def print_top_words(res_idx, vac_idx, sim):
        print(f"Близость: {sim:.4f}")
        print(f"Резюме (первые 100 символов): {processed_resumes[res_idx][:100]}...")
        print(f"Вакансия (первые 100 символов): {processed_vacancies[vac_idx][:100]}...")

        # Получаем векторы
        res_vec = X_res[res_idx].toarray()[0]
        vac_vec = X_vac[vac_idx].toarray()[0]

        # Считаем вклад каждого слова (произведение весов)
        contributions = res_vec * vac_vec

        # Сортируем по убыванию
        top_indices = np.argsort(contributions)[-5:][::-1]

        print("\nТоп-5 слов с наибольшим вкладом:")
        for idx in top_indices:
            if contributions[idx] > 0:
                print(f"  {feature_names[idx]}: {contributions[idx]:.4f}")
        print("-" * 50)

    print("=== ПАРА С САМОЙ ВЫСОКОЙ БЛИЗОСТЬЮ ===")
    print_top_words(max_pair[0], max_pair[1], max_sim)

    print("\n=== ПАРА С САМОЙ НИЗКОЙ БЛИЗОСТЬЮ ===")
    print_top_words(min_pair[0], min_pair[1], min_sim)


if __name__ == "__main__":
    main()
import os
import json
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import read_resume, read_vacancy
from preprocessor import preprocess_text
from build_dataset import rule_for_target

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

    print(f"Вакансий: {len(vacancies)}, Резюме: {len(resumes)}")


    processed_vacancies = [preprocess_text(v) for v in vacancies]
    processed_resumes = [preprocess_text(r) for r in resumes]


    vac_train, vac_test = train_test_split(processed_vacancies, test_size=0.2, random_state=42)

    print(f"Обучающая часть (вакансии): {len(vac_train)}")
    print(f"Проверочная часть (вакансии): {len(vac_test)}")


    vectorizer = CountVectorizer(min_df=2)
    X_train_counts = vectorizer.fit_transform(vac_train)


    vocab_size_before = len(vectorizer.vocabulary_)

    temp_vectorizer = CountVectorizer()
    temp_vectorizer.fit(vac_train)
    vocab_size_total = len(temp_vectorizer.vocabulary_)

    print("\n--- паспорт словаря ---")
    print(f"Размер корпуса: {len(vac_train)} документов (обучающая часть)")
    print(f"Размер словаря до отсечения: {vocab_size_total} слов")
    print(f"Размер словаря после отсечения (min_df=2): {vocab_size_before} слов")


    X_test_counts = vectorizer.transform(vac_test)

    zero_vectors = (X_test_counts.sum(axis=1) == 0).sum()
    print(f"Доля векторов из теста, где нет ни одного знакомого слова: {zero_vectors / len(vac_test) * 100:.1f}%")


if __name__ == "__main__":
    main()
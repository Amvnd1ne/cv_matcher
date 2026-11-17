import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessor import preprocess_text
from data_loader import read_resume
from vectorizer import calculate_vector_score
from build_dataset import rule_for_target

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_DIR = os.path.join(PROJECT_ROOT, 'data', 'corpus')
RESUMES_DIR = os.path.join(PROJECT_ROOT, 'data', 'resumes')


def main():
    # 1. Загружаем корпус (200 вакансий)
    vacancies = []
    for f in os.listdir(CORPUS_DIR):
        if f.endswith('.json'):
            with open(os.path.join(CORPUS_DIR, f), 'r', encoding='utf-8') as file:
                data = json.load(file)
                vacancies.append(data['требования'] + " " + data['обязанности'])

    # 2. Загружаем резюме
    resumes = []
    for f in os.listdir(RESUMES_DIR):
        if f.endswith('.txt'):
            text = read_resume(os.path.join(RESUMES_DIR, f))
            if text:
                resumes.append(text)

    # 3. Создаём пары (первые 50 вакансий * 6 резюме = 300 пар)
    pairs = []
    for res in resumes:
        for vac in vacancies[:50]:
            target = rule_for_target(res, vac)
            pairs.append({
                'resume': res,
                'vacancy': vac,
                'target': target
            })

    df = pd.DataFrame(pairs)

    # 4. Разделение на train/test
    X_text = df[['resume', 'vacancy']]
    y = df['target']

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"Обучающая часть: {len(X_train_text)}")
    print(f"Проверочная часть: {len(X_test_text)}")

    # --- МЕТОД 1: Балл из Занятия 1 ---
    y_pred_z1 = []
    for _, row in X_test_text.iterrows():
        score_data = calculate_vector_score(row['resume'], row['vacancy'])
        y_pred_z1.append(1 if score_data['score'] > 0.3 else 0)

    # --- МЕТОД 2: Признаки из Занятия 2 ---
    def extract_features(row):
        res_len = len(row['resume'])
        vac_len = len(row['vacancy'])
        return [res_len, vac_len, res_len - vac_len]

    X_train_feat = np.array([extract_features(row) for _, row in X_train_text.iterrows()])
    X_test_feat = np.array([extract_features(row) for _, row in X_test_text.iterrows()])

    model_z2 = LogisticRegression(class_weight='balanced', max_iter=1000)
    model_z2.fit(X_train_feat, y_train)
    y_pred_z2 = model_z2.predict(X_test_feat)

    # --- МЕТОД 3: Счётная векторизация + Косинусная близость ---
    # Предобработка отдельных текстов
    train_resumes_processed = X_train_text['resume'].apply(preprocess_text)
    train_vacancies_processed = X_train_text['vacancy'].apply(preprocess_text)
    test_resumes_processed = X_test_text['resume'].apply(preprocess_text)
    test_vacancies_processed = X_test_text['vacancy'].apply(preprocess_text)

    count_vec = CountVectorizer(min_df=2)
    # Обучаем на объединенном словаре (резюме + вакансии)
    all_train_texts = pd.concat([train_resumes_processed, train_vacancies_processed])
    count_vec.fit(all_train_texts)

    # Получаем векторы для train
    X_train_res_count = count_vec.transform(train_resumes_processed)
    X_train_vac_count = count_vec.transform(train_vacancies_processed)

    # Получаем векторы для test
    X_test_res_count = count_vec.transform(test_resumes_processed)
    X_test_vac_count = count_vec.transform(test_vacancies_processed)

    # Считаем косинусную близость для train и test
    train_similarities = [cosine_similarity(X_train_res_count[i], X_train_vac_count[i])[0][0] for i in
                          range(len(train_resumes_processed))]
    test_similarities = [cosine_similarity(X_test_res_count[i], X_test_vac_count[i])[0][0] for i in
                         range(len(test_resumes_processed))]

    # Обучаем модель на косинусной близости (как на единственном признаке)
    model_count = LogisticRegression(class_weight='balanced', max_iter=1000)
    model_count.fit(np.array(train_similarities).reshape(-1, 1), y_train)
    y_pred_count = model_count.predict(np.array(test_similarities).reshape(-1, 1))

    # --- МЕТОД 4: Взвешенная векторизация + Косинусная близость ---
    tfidf_vec = TfidfVectorizer(min_df=2)
    tfidf_vec.fit(all_train_texts)

    X_train_res_tfidf = tfidf_vec.transform(train_resumes_processed)
    X_train_vac_tfidf = tfidf_vec.transform(train_vacancies_processed)
    X_test_res_tfidf = tfidf_vec.transform(test_resumes_processed)
    X_test_vac_tfidf = tfidf_vec.transform(test_vacancies_processed)

    train_similarities_tfidf = [cosine_similarity(X_train_res_tfidf[i], X_train_vac_tfidf[i])[0][0] for i in
                                range(len(train_resumes_processed))]
    test_similarities_tfidf = [cosine_similarity(X_test_res_tfidf[i], X_test_vac_tfidf[i])[0][0] for i in
                               range(len(test_resumes_processed))]

    model_tfidf = LogisticRegression(class_weight='balanced', max_iter=1000)
    model_tfidf.fit(np.array(train_similarities_tfidf).reshape(-1, 1), y_train)
    y_pred_tfidf = model_tfidf.predict(np.array(test_similarities_tfidf).reshape(-1, 1))

    # --- Считаем метрики ---
    def get_metrics(y_true, y_pred, name):
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        return {'Метод': name, 'Precision': prec, 'Recall': rec, 'F1': f1}

    results = [
        get_metrics(y_test, y_pred_z1, 'Балл из Занятия 1'),
        get_metrics(y_test, y_pred_z2, 'Признаки из Занятия 2'),
        get_metrics(y_test, y_pred_count, 'Счётная векторизация (Count)'),
        get_metrics(y_test, y_pred_tfidf, 'Взвешенная векторизация (TF-IDF)')
    ]

    result_df = pd.DataFrame(results)
    print("\n--- СРАВНИТЕЛЬНАЯ ТАБЛИЦА (ИСПРАВЛЕННАЯ) ---")
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
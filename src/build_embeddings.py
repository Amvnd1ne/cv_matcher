import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import time
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import read_resume
from preprocessor import preprocess_text

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_DIR = os.path.join(PROJECT_ROOT, 'data', 'corpus')
RESUMES_DIR = os.path.join(PROJECT_ROOT, 'data', 'resumes')
EMB_DIR = os.path.join(PROJECT_ROOT, 'data', 'embeddings')

os.makedirs(EMB_DIR, exist_ok=True)


def main():
    print("Загрузка модели (может занять минуту)...")
    # Загружаем модель
    model = SentenceTransformer('intfloat/multilingual-e5-small')

    # 1. Загружаем и обрабатываем вакансии
    vacancies = []
    vac_ids = []
    for f in os.listdir(CORPUS_DIR):
        if f.endswith('.json'):
            with open(os.path.join(CORPUS_DIR, f), 'r', encoding='utf-8') as file:
                data = json.load(file)
                # Для e5 модели нужно добавлять префикс "passage: "
                text = "passage: " + data['требования'] + " " + data['обязанности']
                vacancies.append(text)
                vac_ids.append(f)

    # 2. Загружаем и обрабатываем резюме
    resumes = []
    res_ids = []
    for f in os.listdir(RESUMES_DIR):
        if f.endswith('.txt'):
            text = read_resume(os.path.join(RESUMES_DIR, f))
            if text:
                # Для e5 модели нужно добавлять префикс "query: "
                text = "query: " + text
                resumes.append(text)
                res_ids.append(f)

    print(f"Вакансий для кодирования: {len(vacancies)}")
    print(f"Резюме для кодирования: {len(resumes)}")

    # 3. Считаем эмбеддинги вакансий
    start_time = time.time()
    print("Кодирование вакансий...")
    vac_embeddings = model.encode(vacancies, show_progress_bar=True)
    vac_time = time.time() - start_time
    print(f"Время кодирования вакансий: {vac_time:.2f} сек")

    # 4. Считаем эмбеддинги резюме
    start_time = time.time()
    print("Кодирование резюме...")
    res_embeddings = model.encode(resumes, show_progress_bar=True)
    res_time = time.time() - start_time
    print(f"Время кодирования резюме: {res_time:.2f} сек")

    # 5. Сохраняем на диск
    np.save(os.path.join(EMB_DIR, 'vac_embeddings.npy'), vac_embeddings)
    np.save(os.path.join(EMB_DIR, 'res_embeddings.npy'), res_embeddings)

    with open(os.path.join(EMB_DIR, 'vac_ids.json'), 'w', encoding='utf-8') as f:
        json.dump(vac_ids, f, ensure_ascii=False)

    with open(os.path.join(EMB_DIR, 'res_ids.json'), 'w', encoding='utf-8') as f:
        json.dump(res_ids, f, ensure_ascii=False)

    print(f"\nЭмбеддинги сохранены в {EMB_DIR}")
    print(f"Размерность вектора: {vac_embeddings.shape[1]}")


if __name__ == "__main__":
    main()
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

# Списки пар (имя файла резюме, имя файла вакансии)
POSITIVE_PAIRS = [
    ('resume_strong.txt', 'vac_07.json'),
    ('resume_strong.txt', 'vac_04.json'),
    ('resume_strong.txt', 'vac_09.json'),
    ('resume_strong2.txt', 'vac_07.json'),
    ('resume_strong2.txt', 'vac_03.json'),
    ('resume_related.txt', 'vac_03.json'),
    ('resume_related.txt', 'vac_04.json'),
    ('resume_related2.txt', 'vac_05.json'),
    ('resume_related2.txt', 'vac_08.json'),
    ('resume_strong.txt', 'vac_15.json'),
    # Добавьте еще 10 пар, чтобы было 20
]

NEGATIVE_PAIRS = [
    ('resume_weak.txt', 'vac_07.json'),
    ('resume_weak.txt', 'vac_04.json'),
    ('resume_weak2.txt', 'vac_01.json'),
    ('resume_strong.txt', 'vac_01.json'),
    ('resume_strong2.txt', 'vac_13.json'),
    ('resume_related.txt', 'vac_01.json'),
    ('resume_related2.txt', 'vac_10.json'),
    ('resume_weak.txt', 'vac_11.json'),
    # Добавьте еще 12 пар, чтобы было 20
]


def main():
    print("Загрузка модели...")
    model = SentenceTransformer('intfloat/multilingual-e5-small')

    def get_embedding(text, prefix):
        return model.encode(prefix + text)

    # Предзагружаем тексты
    res_texts = {}
    for f in os.listdir(RESUMES_DIR):
        if f.endswith('.txt'):
            res_texts[f] = read_resume(os.path.join(RESUMES_DIR, f))

    vac_texts = {}
    for f in os.listdir(VACANCIES_DIR):
        if f.endswith('.json'):
            data = read_vacancy(os.path.join(VACANCIES_DIR, f))
            if data:
                vac_texts[f] = data['требования'] + " " + data['обязанности']

    def calc_group(pairs):
        scores = []
        for res_name, vac_name in pairs:
            if res_name not in res_texts or vac_name not in vac_texts:
                continue

            res_emb = get_embedding(res_texts[res_name], "query: ").reshape(1, -1)
            vac_emb = get_embedding(vac_texts[vac_name], "passage: ").reshape(1, -1)

            sim = cosine_similarity(res_emb, vac_emb)[0][0]
            scores.append(sim)
        return np.array(scores)

    print("\nРасчет положительных пар...")
    pos_scores = calc_group(POSITIVE_PAIRS)
    print(f"Средняя близость (положительные): {np.mean(pos_scores):.3f}")

    print("\nРасчет отрицательных пар...")
    neg_scores = calc_group(NEGATIVE_PAIRS)
    print(f"Средняя близость (отрицательные): {np.mean(neg_scores):.3f}")

    gap = np.mean(pos_scores) - np.mean(neg_scores)
    print(f"\nРазрыв: {gap:.3f}")

    if gap > 0.1:
        print("Разрыв больше 0.1 — модель работает хорошо!")
    else:
        print("Разрыв меньше 0.1 — модель не различает пары.")


if __name__ == "__main__":
    main()
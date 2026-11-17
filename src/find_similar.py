import os
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMB_DIR = os.path.join(PROJECT_ROOT, 'data', 'embeddings')
CORPUS_DIR = os.path.join(PROJECT_ROOT, 'data', 'corpus')


def main():
    # 1. Загружаем эмбеддинги
    res_emb = np.load(os.path.join(EMB_DIR, 'res_embeddings.npy'))
    vac_emb = np.load(os.path.join(EMB_DIR, 'vac_embeddings.npy'))

    with open(os.path.join(EMB_DIR, 'res_ids.json'), 'r', encoding='utf-8') as f:
        res_ids = json.load(f)
    with open(os.path.join(EMB_DIR, 'vac_ids.json'), 'r', encoding='utf-8') as f:
        vac_ids = json.load(f)

    print(f"Загружено резюме: {len(res_ids)}, вакансий: {len(vac_ids)}")

    # 2. Считаем косинусную близость (матрица 6x200)
    sim_matrix = cosine_similarity(res_emb, vac_emb)

    # 3. Для каждого резюме находим топ-5 ближайших вакансий
    for i, res_id in enumerate(res_ids):
        print(f"\n--- Резюме: {res_id} ---")
        # Сортируем индексы по убыванию близости
        top_indices = np.argsort(sim_matrix[i])[::-1][:5]

        for idx in top_indices:
            vac_id = vac_ids[idx]
            score = sim_matrix[i][idx]

            # Читаем название вакансии для наглядности
            with open(os.path.join(CORPUS_DIR, vac_id), 'r', encoding='utf-8') as f:
                vac_data = json.load(f)
                title = vac_data['должность']

            print(f"  {score:.3f} | {vac_id} | {title}")


if __name__ == "__main__":
    main()
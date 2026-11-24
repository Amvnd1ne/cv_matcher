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
VACANCIES_DIR = os.path.join(PROJECT_ROOT, 'data', 'corpus')
EMB_DIR = os.path.join(PROJECT_ROOT, 'data', 'embeddings')


def find_relevant_fragments(resume_text, missing_words, top_k=20):
    """
    Находит топ-k вакансий, похожих на резюме, и вырезает из них
    предложения, содержащие недостающие слова.
    """
    model = SentenceTransformer('intfloat/multilingual-e5-small')
    vac_emb = np.load(os.path.join(EMB_DIR, 'vac_embeddings.npy'))

    with open(os.path.join(EMB_DIR, 'vac_ids.json'), 'r', encoding='utf-8') as f:
        vac_ids = json.load(f)

    res_emb = model.encode("query: " + resume_text).reshape(1, -1)
    sim_matrix = cosine_similarity(res_emb, vac_emb)[0]
    top_indices = np.argsort(sim_matrix)[::-1][:top_k]

    fragments = []
    seen_texts = set()  # Храним уникальные тексты фрагментов

    for idx in top_indices:
        vac_id = vac_ids[idx]
        with open(os.path.join(VACANCIES_DIR, vac_id), 'r', encoding='utf-8') as f:
            vac_data = json.load(f)
            vac_text = vac_data['требования'] + " " + vac_data['обязанности']

        sentences = vac_text.split('.')
        for sentence in sentences:
            for word in missing_words:
                if word.lower() in sentence.lower():
                    frag = f"Из вакансии '{vac_data['должность']}': {sentence.strip()}."
                    if frag not in seen_texts:
                        fragments.append(frag)
                        seen_texts.add(frag)
                    break
            if len(fragments) >= 5:
                break

        if len(fragments) >= 5:
            break

    return fragments


def main():
    # Тестовые данные (реальные для нашего корпуса аналитиков)
    resume_text = "Опыт работы 3 года. Знаю Python, SQL, Excel."
    missing_words = ["sql", "python", "excel", "отчетность"]  # <-- ВОТ ЗДЕСЬ

    print("--- Поиск релевантных фрагментов (RAG) ---")
    fragments = find_relevant_fragments(resume_text, missing_words)

    if fragments:
        print(f"Найдено фрагментов: {len(fragments)}")
        for i, frag in enumerate(fragments, 1):
            print(f"{i}. {frag}")
    else:
        print("Фрагменты не найдены (возможно, недостающие слова не встречаются в других вакансиях).")

if __name__ == "__main__":
    main()
import os
import json
import math
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessor import preprocess_text

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_DIR = os.path.join(PROJECT_ROOT, 'data', 'corpus')


def main():

    vacancies = []
    for f in os.listdir(CORPUS_DIR):
        if f.endswith('.json'):
            with open(os.path.join(CORPUS_DIR, f), 'r', encoding='utf-8') as file:
                data = json.load(file)
                vacancies.append(data['требования'] + " " + data['обязанности'])


    processed = [preprocess_text(v) for v in vacancies]


    all_words = " ".join(processed).split()
    word_counts = pd.Series(all_words).value_counts()


    top_words = word_counts.index.tolist()
    word_frequent = top_words[0]
    word_medium = top_words[len(top_words) // 2]  # Среднее
    word_rare = top_words[-1]

    words_to_check = [word_frequent, word_medium, word_rare]

    print("--- ручной расчёт TF-IDF ---")
    N = len(processed)


    manual_results = {}
    for word in words_to_check:

        doc = processed[0].split()
        tf = doc.count(word) / len(doc) if len(doc) > 0 else 0


        df = sum(1 for d in processed if word in d.split())
        idf = math.log(N / df) if df > 0 else 0

        tf_idf = tf * idf
        manual_results[word] = tf_idf

        print(f"Слово: '{word}'")
        print(f"  Встречается в {df} документах из {N}")
        print(f"  TF (в первом документе): {tf:.4f}")
        print(f"  IDF: {idf:.4f}")
        print(f"  TF-IDF (вручную): {tf_idf:.4f}")
        print()

    print("--- расчёт библиотекой TfidfVectorizer ---")
    vectorizer = TfidfVectorizer(min_df=2)
    X = vectorizer.fit_transform(processed)

    feature_names = vectorizer.get_feature_names_out()

    word_to_index = {word: i for i, word in enumerate(feature_names)}

    for word in words_to_check:
        if word in word_to_index:
            idx = word_to_index[word]

            lib_value = X[0, idx]
            manual_value = manual_results[word]
            diff = abs(lib_value - manual_value)
            print(f"Слово: '{word}'")
            print(f"  Вручную: {manual_value:.4f}")
            print(f"  Библиотека: {lib_value:.4f}")
            print(f"  Разница: {diff:.4f} (объясняется сглаживанием в формуле библиотеки)")
            print()
        else:
            print(f"Слово '{word}' не попало в словарь библиотеки (редкое).")


if __name__ == "__main__":
    main()
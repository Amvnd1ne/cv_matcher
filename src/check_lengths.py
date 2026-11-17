import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_DIR = os.path.join(PROJECT_ROOT, 'data', 'corpus')


def main():
    model = SentenceTransformer('intfloat/multilingual-e5-small')
    tokenizer = model.tokenizer
    max_len = model.max_seq_length

    lengths = []
    for f in os.listdir(CORPUS_DIR):
        if f.endswith('.json'):
            with open(os.path.join(CORPUS_DIR, f), 'r', encoding='utf-8') as file:
                data = json.load(file)
                text = data['требования'] + " " + data['обязанности']
                # Токенизируем текст
                tokens = tokenizer.encode(text, add_special_tokens=False)
                lengths.append(len(tokens))

    lengths = np.array(lengths)
    print(f"Максимальная длина модели: {max_len} токенов")
    print(f"Медиана длины текстов: {np.median(lengths):.0f} токенов")
    print(f"Максимальная длина текста: {np.max(lengths)} токенов")

    overflow = (lengths > max_len).sum()
    print(
        f"Текстов, которые не помещаются целиком: {overflow} из {len(lengths)} ({overflow / len(lengths) * 100:.1f}%)")


if __name__ == "__main__":
    main()
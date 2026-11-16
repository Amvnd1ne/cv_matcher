from collections import Counter
from text_processor import normalize_text


def calculate_vector_score(resume_text: str, vacancy_text: str) -> dict:
    """
    Считает балл соответствия через векторный подход (скалярное произведение).
    Возвращает словарь с результатами:
    - score: нормированный балл (от 0 до 1)
    - dot_product: скалярное произведение
    - resume_vector: вектор частот резюме
    - vacancy_vector: вектор частот вакансии
    - common_words: общий словарь (все уникальные слова)
    """

    resume_words = normalize_text(resume_text)
    vacancy_words = normalize_text(vacancy_text)


    common_words = set(resume_words).union(set(vacancy_words))


    resume_counter = Counter(resume_words)
    vacancy_counter = Counter(vacancy_words)


    resume_vector = [resume_counter.get(word, 0) for word in common_words]
    vacancy_vector = [vacancy_counter.get(word, 0) for word in common_words]


    dot_product = sum(a * b for a, b in zip(resume_vector, vacancy_vector))

    max_possible = sum(a * a for a in vacancy_vector)
    if max_possible == 0:
        score = 0.0
    else:
        score = dot_product / max_possible

    return {
        "score": score,
        "dot_product": dot_product,
        "resume_vector": resume_vector,
        "vacancy_vector": vacancy_vector,
        "common_words": list(common_words)
    }


def calculate_binary_vector_score(resume_text: str, vacancy_text: str) -> dict:
    """
    Считает балл на двоичных векторах (1 - слово есть, 0 - нет).
    Результат должен совпасть с числом общих слов.
    """
    resume_words = set(normalize_text(resume_text))
    vacancy_words = set(normalize_text(vacancy_text))

    common_words = resume_words.intersection(vacancy_words)

    common_dict = list(resume_words.union(vacancy_words))
    resume_binary = [1 if word in resume_words else 0 for word in common_dict]
    vacancy_binary = [1 if word in vacancy_words else 0 for word in common_dict]

    dot_product = sum(a * b for a, b in zip(resume_binary, vacancy_binary))

    return {
        "dot_product": dot_product,
        "common_words_count": len(common_words),
        "common_words": list(common_words)
    }
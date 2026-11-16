from text_processor import normalize_text


def calculate_overlap(resume_text: str, vacancy_text: str) -> dict:
    """
    Сравнивает текст резюме и текст вакансии.
    Возвращает словарь с результатами:
    - score: доля покрытия требований (от 0 до 1)
    - matched_words: список совпавших слов
    - missing_words: список слов из вакансии, которых нет в резюме
    """

    resume_words = normalize_text(resume_text)
    vacancy_words = normalize_text(vacancy_text)

    resume_set = set(resume_words)
    vacancy_set = set(vacancy_words)

    matched_words = list(resume_set.intersection(vacancy_set))

    missing_words = list(vacancy_set.difference(resume_set))

    if len(vacancy_set) == 0:
        score = 0.0
    else:
        score = len(matched_words) / len(vacancy_set)

    return {
        "score": score,
        "matched_words": matched_words,
        "missing_words": missing_words
    }
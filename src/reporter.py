def format_explanation(score: float, matched_words: list, missing_words: list, vacancy_title: str = "") -> str:
    """
    Формирует объяснение для кандидата в формате из ТЗ.
    """

    matched_top = matched_words[:10]
    missing_top = missing_words[:10]

    lines = []
    if vacancy_title:
        lines.append(f"Вакансия: {vacancy_title}")
    lines.append(f"Балл соответствия: {score:.2f}")
    lines.append("")

    if matched_top:
        lines.append("Совпало:")
        lines.append(", ".join(matched_top))
    else:
        lines.append("Совпало: (нет совпадений)")
    lines.append("")

    if missing_top:
        lines.append("Не хватает:")
        lines.append(", ".join(missing_top))
    else:
        lines.append("Не хватает: (все ключевые слова найдены)")
    lines.append("")

    lines.append("Оценка посчитана по совпадению слов и не учитывает, что одно и то же может быть названо по-разному.")

    return "\n".join(lines)
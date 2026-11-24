import os
import json
import time
import random
from pydantic import BaseModel, Field
from typing import List
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import read_resume, read_vacancy
from vectorizer import calculate_vector_score
from matcher import calculate_overlap
from rag_agent import find_relevant_fragments

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESUMES_DIR = os.path.join(PROJECT_ROOT, 'data', 'resumes')
VACANCIES_DIR = os.path.join(PROJECT_ROOT, 'data', 'vacancies')


# --- Схема ответа (Pydantic) ---
class RecommendationResponse(BaseModel):
    analysis: str = Field(..., min_length=10, max_length=500)
    advice: List[str] = Field(..., min_length=1, max_length=5)


# --- Имитация вызова LLM с обработкой ошибок ---
def call_llm_with_retry(prompt, max_retries=3):
    for attempt in range(1, max_retries + 1):
        try:
            # Имитация случайной ошибки (20% шанс)
            if random.random() < 0.2:
                raise TimeoutError("Сервис не ответил вовремя")

            # Успешный ответ (заглушка, имитирующая LLM)
            return {
                "analysis": "На основе анализа резюме и вакансии, а также примеров из похожих вакансий, можно сделать вывод о частичном соответствии.",
                "advice": [
                    "Добавьте в резюме конкретные примеры использования SQL и Python.",
                    "Используйте формулировки из примеров (например, 'SQL: JOIN, агрегации'), чтобы описать свой опыт.",
                    "Укажите, с какими BI-инструментами вы работали."
                ]
            }
        except TimeoutError as e:
            print(f"⚠️ Попытка {attempt}/{max_retries}: {e}")
            if attempt < max_retries:
                wait_time = 2 ** attempt
                print(f"   Ждем {wait_time} сек...")
                time.sleep(wait_time)
            else:
                print("❌ Все попытки исчерпаны. Возвращаем запасной ответ.")
                return {
                    "analysis": "Не удалось получить рекомендации от сервиса. Попробуйте позже.",
                    "advice": ["Проверьте резюме вручную на соответствие требованиям вакансии."]
                }


def get_recommendations(resume_text, vacancy_text, score, matched_words, missing_words):
    # 1. RAG: находим фрагменты
    print("🔍 Поиск релевантных фрагментов (RAG)...")
    fragments = find_relevant_fragments(resume_text, missing_words)
    print(f"   Найдено фрагментов: {len(fragments)}")

    # 2. Формируем промпт (в реальности он бы ушел в LLM)
    fragments_text = "\n".join(fragments) if fragments else "Фрагменты не найдены."
    prompt = f"""
    Резюме: {resume_text[:300]}...
    Вакансия: {vacancy_text[:300]}...
    Балл: {score:.2f}
    Не хватает: {', '.join(missing_words)}
    Примеры из других вакансий:
    {fragments_text}
    """

    # 3. Вызываем LLM (с обработкой ошибок)
    print("🤖 Вызов LLM (имитация)...")
    raw_response = call_llm_with_retry(prompt)

    # 4. Валидируем ответ
    try:
        validated = RecommendationResponse(**raw_response)
        return validated.model_dump()
    except Exception as e:
        print(f"⚠️ Ошибка валидации: {e}")
        return {
            "analysis": "Не удалось получить корректный ответ от модели.",
            "advice": ["Попробуйте позже."]
        }


def main():
    print("=== AI-Агент рекомендаций ===")

    # Загружаем данные
    resume_text = read_resume(os.path.join(RESUMES_DIR, 'resume_strong.txt'))
    vacancy_data = read_vacancy(os.path.join(VACANCIES_DIR, 'vac_07.json'))

    if not resume_text or not vacancy_data:
        print("❌ Ошибка загрузки данных.")
        return

    vacancy_text = vacancy_data['требования'] + " " + vacancy_data['обязанности']

    # Считаем балл (из Занятия 1)
    score_data = calculate_vector_score(resume_text, vacancy_text)
    score = score_data['score']

    # Находим недостающие слова
    overlap = calculate_overlap(resume_text, vacancy_text)
    missing_words = overlap['missing_words'][:10]  # Берем первые 10

    print(f"Вакансия: {vacancy_data['должность']}")
    print(f"Балл: {score:.2f}")
    print(f"Не хватает: {', '.join(missing_words[:5])}...")
    print("-" * 50)

    # Получаем рекомендации
    result = get_recommendations(resume_text, vacancy_text, score, overlap['matched_words'], missing_words)

    # Выводим результат
    print("\n=== РЕКОМЕНДАЦИИ ===")
    print(f"Анализ: {result['analysis']}")
    print("Советы:")
    for i, advice in enumerate(result['advice'], 1):
        print(f"  {i}. {advice}")


if __name__ == "__main__":
    main()
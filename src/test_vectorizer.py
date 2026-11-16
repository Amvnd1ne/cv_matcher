import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import read_resume, read_vacancy
from vectorizer import calculate_vector_score, calculate_binary_vector_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RESUMES_DIR = os.path.join(PROJECT_ROOT, 'data', 'resumes')
VACANCIES_DIR = os.path.join(PROJECT_ROOT, 'data', 'vacancies')

print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"RESUMES_DIR: {RESUMES_DIR}")
print(f"VACANCIES_DIR: {VACANCIES_DIR}")
print("-" * 50)


def main():
    print("--- Тест векторного подхода ---")

    resume_path = os.path.join(RESUMES_DIR, 'resume_strong.txt')
    vacancy_path = os.path.join(VACANCIES_DIR, 'vac_01.json')

    resume_text = read_resume(resume_path)
    vacancy_data = read_vacancy(vacancy_path)

    if not resume_text:
        print("Не удалось прочитать резюме.")
        return
    if not vacancy_data:
        print("Не удалось прочитать вакансию.")
        return

    vacancy_text = vacancy_data['требования'] + " " + vacancy_data['обязанности']

    print(f"\nВакансия: {vacancy_data['должность']}")
    print(f"Длина резюме: {len(resume_text)} символов")
    print(f"Длина текста вакансии: {len(vacancy_text)} символов")

    print("\n1. Вектор частот (скалярное произведение):")
    result = calculate_vector_score(resume_text, vacancy_text)
    print(f"   Балл: {result['score']:.4f}")
    print(f"   Скалярное произведение: {result['dot_product']}")

    print("\n2. Двоичный вектор:")
    binary_result = calculate_binary_vector_score(resume_text, vacancy_text)
    print(f"   Скалярное произведение: {binary_result['dot_product']}")
    print(f"   Число общих слов: {binary_result['common_words_count']}")

    if binary_result['dot_product'] == binary_result['common_words_count']:
        print("\n   Совпадает с числом общих слов из Шага 5!")
    else:
        print("\n   Не совпадает! Проверьте код.")

    print(f"\nОбщие слова ({len(binary_result['common_words'])}):")
    print(", ".join(binary_result['common_words']))


if __name__ == "__main__":
    main()
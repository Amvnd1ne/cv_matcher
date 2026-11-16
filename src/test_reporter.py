import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import read_resume, read_vacancy
from vectorizer import calculate_vector_score
from matcher import calculate_overlap
from reporter import format_explanation

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESUMES_DIR = os.path.join(PROJECT_ROOT, 'data', 'resumes')
VACANCIES_DIR = os.path.join(PROJECT_ROOT, 'data', 'vacancies')


def main():
    # Читаем сильное резюме и вакансию Junior продуктовый аналитик Wink
    resume_text = read_resume(os.path.join(RESUMES_DIR, 'resume_strong.txt'))
    vacancy_data = read_vacancy(os.path.join(VACANCIES_DIR, 'vac_07.json'))

    if resume_text and vacancy_data:
        vacancy_text = vacancy_data['требования'] + " " + vacancy_data['обязанности']

        # Считаем балл (векторный подход)
        result = calculate_vector_score(resume_text, vacancy_text)

        # Получаем списки слов (из matcher, где мы считали пересечение)
        overlap_result = calculate_overlap(resume_text, vacancy_text)

        # Формируем вывод
        explanation = format_explanation(
            score=result['score'],
            matched_words=overlap_result['matched_words'],
            missing_words=overlap_result['missing_words'],
            vacancy_title=vacancy_data['должность']
        )

        print(explanation)
        print("\n" + "=" * 50)
        print("Конец вывода")


if __name__ == "__main__":
    main()
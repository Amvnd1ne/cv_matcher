import os
from data_loader import read_resume, read_vacancy
from matcher import calculate_overlap

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESUMES_DIR = os.path.join(BASE_DIR, 'data', 'resumes')
VACANCIES_DIR = os.path.join(BASE_DIR, 'data', 'vacancies')


def main():
    print("--- Тест сравнения резюме и вакансии ---")

    resume_path = os.path.join(RESUMES_DIR, 'resume_strong.txt')
    resume_text = read_resume(resume_path)

    vac_path = os.path.join(VACANCIES_DIR, 'vac_01.json')
    vacancy_data = read_vacancy(vac_path)

    if resume_text and vacancy_data:

        vacancy_text = vacancy_data['требования'] + " " + vacancy_data['обязанности']

        result = calculate_overlap(resume_text, vacancy_text)

        print(f"\nВакансия: {vacancy_data['должность']}")
        print(f"Балл соответствия: {result['score']:.2f}")
        print(f"\nСовпавшие слова ({len(result['matched_words'])}):")
        print(", ".join(result['matched_words']))
        print(f"\nНедостающие слова (первые 10 из {len(result['missing_words'])}):")
        print(", ".join(result['missing_words'][:10]))
    else:
        print("Не удалось прочитать файлы. Проверьте пути.")


if __name__ == "__main__":
    main()
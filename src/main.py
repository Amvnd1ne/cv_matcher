import os
import json
from data_loader import read_resume, read_vacancy

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VACANCIES_DIR = os.path.join(BASE_DIR, 'data', 'vacancies')
RESUMES_DIR = os.path.join(BASE_DIR, 'data', 'resumes')


def main():
    print("--- Проверка чтения резюме ---")
    resume_path = os.path.join(RESUMES_DIR, 'resume_strong.txt')
    text = read_resume(resume_path)
    if text:
        print(f"Успешно прочитано резюме. Длина {len(text)} символов.")

    print("\n--- Проверка чтения вакансии ---")
    vac_path = os.path.join(VACANCIES_DIR, 'vac_01.json')
    vacancy = read_vacancy(vac_path)
    if vacancy:
        print(f"Успешно прочитана вакансия {vacancy.get('должность')}")
        print(f"   Требования {vacancy.get('требования')[:50]}...")

    print("\n--- Проверка обработки ошибок (Битые файлы) ---")
    print("Тест 1: Несуществующий файл")
    read_resume("fake_file.txt")
    read_vacancy("fake_vac.json")

    print("\nТест 2: JSON без поля 'требования'")
    bad_json_path = os.path.join(VACANCIES_DIR, 'bad_vac.json')
    with open(bad_json_path, 'w', encoding='utf-8') as f:
        json.dump({"должность": "Тест"}, f)
    read_vacancy(bad_json_path)
    os.remove(bad_json_path)

from text_processor import normalize_text


test_text = "Python-разработчик, опыт работы с Django и PostgreSQL! Знание SQL, Git."

result = normalize_text(test_text)

print("Исходный текст:", test_text)
print("Очищенные слова:", result)
if __name__ == "__main__":
    main()
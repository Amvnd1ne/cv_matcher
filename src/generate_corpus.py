import os
import json
import random
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import read_vacancy

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VACANCIES_DIR = os.path.join(PROJECT_ROOT, 'data', 'vacancies')
CORPUS_DIR = os.path.join(PROJECT_ROOT, 'data', 'corpus')


os.makedirs(CORPUS_DIR, exist_ok=True)


def generate_corpus(num_vacancies=200):

    base_vacancies = []
    for f in os.listdir(VACANCIES_DIR):
        if f.endswith('.json'):
            data = read_vacancy(os.path.join(VACANCIES_DIR, f))
            if data:
                base_vacancies.append(data)

    print(f"Загружено базовых вакансий: {len(base_vacancies)}")


    for i in range(1, num_vacancies + 1):
        base = random.choice(base_vacancies)

        noise = random.choice([
            "Требуется опыт работы в команде.",
            "Гибкий график, ДМС.",
            "Работа в офисе в центре Москвы.",
            "Возможны командировки.",
            "Оформление по ТК РФ.",
            "Дружный коллектив и интересные задачи."
        ])

        new_vacancy = {
            "должность": base["должность"],
            "требования": base["требования"] + " " + noise,
            "обязанности": base["обязанности"]
        }

        filename = f"vac_gen_{i:03d}.json"
        filepath = os.path.join(CORPUS_DIR, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(new_vacancy, f, ensure_ascii=False, indent=2)

    print(f"Сгенерировано {num_vacancies} вакансий в папку {CORPUS_DIR}")


if __name__ == "__main__":
    generate_corpus(200)
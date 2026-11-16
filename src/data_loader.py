import json


def read_resume(file_path: str) -> str | None:
    """
    Читает текстовый файл (.txt) с резюме.
    Возвращает текст или None, если произошла ошибка.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text
    except FileNotFoundError:
        print(f"Ошибка Файл не найден по пути {file_path}")
        return None
    except UnicodeDecodeError:
        print(f"Ошибка кодировки Файл {file_path} не в кодировке UTF-8")
        return None
    except Exception as e:
        print(f"Неизвестная ошибка при чтении {file_path}: {e}")
        return None


def read_vacancy(file_path: str) -> dict | None:
    """
    Читает JSON файл с вакансией и проверяет наличие нужных полей.
    Возвращает словарь или None, если произошла ошибка или нет полей.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        required_fields = ['должность', 'требования', 'обязанности']
        for field in required_fields:
            if field not in data:
                print(f"Ошибка В файле {file_path} отсутствует поле '{field}'")
                return None

        return data
    except FileNotFoundError:
        print(f"Ошибка Файл не найден по пути {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Ошибка Файл {file_path} не является корректным JSON")
        return None
    except Exception as e:
        print(f"Неизвестная ошибка при чтении {file_path}: {e}")
        return None
import re

BASE_STOP_WORDS = {
    "и", "в", "на", "с", "по", "для", "от", "до", "из", "к", "у", "о", "об",
    "за", "под", "над", "при", "без", "через", "между", "или", "но", "а",
    "что", "чтобы", "как", "так", "это", "весь", "все", "всех", "свой",
    "свои", "который", "которая", "которые", "которое", "быть", "был",
    "была", "было", "были", "есть", "не", "ни", "же", "ли", "бы", "мы",
    "вы", "он", "она", "они", "оно", "я", "ты", "ваш", "наш", "их", "его", "ее"
}


DOMAIN_STOP_WORDS = {
    "компания", "работа", "задача", "требование", "обязанность", "знание",
    "опыт", "умение", "навык", "вакансия", "должность", "условие", "график",
    "офис", "зарплата", "коллектив", "развитие", "рост", "команда"
}


ALL_STOP_WORDS = BASE_STOP_WORDS.union(DOMAIN_STOP_WORDS)


def preprocess_text(text: str) -> str:
    """
    Конвейер предобработки (6 шагов).
    Возвращает строку очищенных слов, разделенных пробелом.
    """
    if not text:
        return ""


    text = text.lower().strip()


    text = re.sub(r'[^\w\s]', ' ', text)


    words = text.split()


    words = [w for w in words if len(w) > 2]


    words = [w for w in words if w not in ALL_STOP_WORDS]


    def simple_stem(word):

        for suffix in ['ая', 'ое', 'ые', 'ый', 'ий', 'ов', 'ев', 'ах', 'ям', 'ами', 'ться', 'тся', 'ешь', 'ишь', 'ет',
                       'ит', 'ют', 'ут']:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return word[:-len(suffix)]
        return word

    words = [simple_stem(w) for w in words]

    return " ".join(words)


def get_stop_words_list():
    """Возвращает список стоп-слов (для паспорта)"""
    return sorted(list(ALL_STOP_WORDS))
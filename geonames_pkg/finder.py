# файл с классом поиска городов
# базовые импорты
import json
import os
import numpy as np

# импорты для работы векторами
import torch
from sentence_transformers import SentenceTransformer, util

# импорты для коррекции ошибок
from fuzzywuzzy import process
from transliterate import translit
from yaspeller import check

RANDOM = 12345
torch.manual_seed(RANDOM)
np.random.seed(RANDOM)


class FindCity:
    """
    Класс FindCity для поиска города по векторному представлению.
    """

    def __init__(self, model_id=None, device="cpu", dataset=None, emb_col=None, cols_output=None):
        """
        Инициализация объекта класса FindCity для поиска города.

         Параметры::
            model_id (str): имя модели для векторизации, по умолчанию равно None,
            device (str): акселератор для создания векторов CPU или GPU, по
                          умолчанию равно 'cpu',
            dataset (pd.DataFrame): датасет, содержащий векторные представления городов,
                                    по которому будет осуществляться поиск, по
                                    умолчанию равно None,
            emb_col (str): наименование столбца с векторами представлений, по
                           умолчанию равно None,
            cols_output (list): список наименований столбцов для вывода результата, по
                                умолчанию равно None.
        """
        self.model_id = model_id
        self.device = device
        self.dataset = dataset
        self.emb_col = emb_col
        self.cities_emb = np.array(list(self.dataset[self.emb_col]), dtype=np.float32)
        self.model = SentenceTransformer(self.model_id, device=self.device)
        self.cols_output = cols_output

    @staticmethod
    def spell_checker(city=None):
        """
        Статический метод spell_checker класса FindCity.
        Проверка названия города для исправления возможных опечаток.
        Метод для первичной проверки опечаток, использует метод для
        проверки правописания из пакета yaspeller (Яндекс Спеллер)

         Параметры:
             city (str): название города для проверки, по умолчанию равно None.

         Возвращаемое значение:
             city (str): скорректированное название города или исходное значение,
                         в случае невозможности корректировки.
         """
        # в переменную res записывается True или False.
        # вызывается метод check из yaspeller, если нету ошибок res = True,
        # в противном случае res = False
        res = check(city, lang="ru")
        # если True
        if res.is_ok:
            # возврат исходного слова.
            return city
        # если False
        else:
            # с помощью метода first_match находим первое близкое правильное значение.
            city = res.first_match()
            # возврат исправленного названия города.
            return city

    @staticmethod
    def advanced_spell_checker(city=None, dataset=None):
        """
        Статический метод advanced_spell_checker класса FindCity.
        Расширенная проверка названия города в случае невозможности исправить
        первым методом spell_checker.
        Метод ищет совпадения заданного города в альтернативных именах. Если
        совпадение найдено, возвращается имя города из поля name, если нет,
        то возвращается исходное значение. В процессе используются методы:
         - translit библиотеки transliterate,
         - process библиотеки fuzzywuzzy.
        В основном расширенная проверка нужна для сокращений:
         - МСК - Москва,
         - СПБ - Санкт-Петербург.
         Параметры:
              city (str): название города для проверки, по умолчанию равно None.
              dataset (pd.DataFrame): датасет с городами и альтернативными именами городов,
                                      по умолчанию равно None.
         Возвращаемое значение:
              city (str): скорректированное название города или исходное значение,
                        в случае невозможности корректировки.
        """
        # создаем словарь из датасета, где ключ это значение из поля name, а значения
        # это строка альтернативных имен.
        cities_dict = (
            dataset.groupby("name")["alternatenames"]
            .apply(lambda x: ",".join(map(str, x)))
            .to_dict()
        )
        # транслитерация введенного на русском языке названия города
        city_t = translit(city, "ru", reversed=True)
        # пустой список для найденных городов
        key_lst = []
        # цикл по словарю
        for key, value in cities_dict.items():
            # проверка наличия имени города в исходном или транслитном значении в разбитой на список строке
            # c альтернативными названиями
            if (city.lower() in value.lower().split(", ")) or (
                    city_t.lower() in value.lower().split(", ")
            ):
                # если поиск успешен добавляем ключ - имя города в список
                key_lst.append(key)
            # если поиск безрезультатен
            else:
                # проходим циклом по списку из разделенной строки с альтернативными названиями
                for item in value.lower().split(", "):
                    # транслитерация названия
                    item_t = translit(item, "ru", reversed=True)
                    # альтернативное имя начинается с искомого слова в оригинале или транслите
                    if item.startswith(city) or item_t.startswith(city_t):
                        # то добавляем ключ в список имен
                        key_lst.append(key)
        # в списке может быть много городов и дублей городов, для этого через множество удалим дубли
        key_lst = list(set(key_lst))
        # если длина списка 1 значение
        if len(key_lst) == 1:
            # то в возвращаемую переменную присваиваем 0-й элемент списка
            city = key_lst[0]
            # возвращаемое значение
            return city
        # если городов в списке больше двух
        elif len(key_lst) > 1:
            # то методом process.extract библиотеки fuzzywuzzy вытаскиваем наиболее близкий по расстоянию Левенштейна
            word = process.extractOne(city_t, key_lst)
            # т.к. в word записан список кортежей, то берем из него только название города
            city = word[0][0]
            # возвращаемое значение
            return city
        # если ничего не найдено
        else:
            city = translit(city.lower(), "ru", reversed=True)
            # то возвращаем транслитное значение введенного слова
            return city

    def get_city(
            self,
            city=None,
            top_k=1,
            adv_spell_check=False,
            output_dict_json=False,
            save_json_file=False,
            work_dir=None,
    ):
        """
        Получение информации о городе на основе введенного названия.

         Параметры:
            city (str): название города для поиска, по умолчанию равно None,
            top_k (int): количество наиболее похожих городов для вывода, по умолчанию равно 1,
            adv_spell_check (bool): флаг использования расширенной проверки названия города,
                                    по умолчанию равно False,
            output_dict_json (bool): флаг вывода результата в формате JSON, по умолчанию равно False,
            save_json_file (bool): флаг сохранения результата в JSON-файл, по умолчанию равно False,
            work_dir (str): каталог для сохранения JSON-файла, по умолчанию равно None/

         Возвращаемое значение:
            result_df (pd.DataFrame): если вывод таблицей,
            output_dict (dict): если вывод словарём.
                    """
        # первичная проверка на исправление ошибок
        city = FindCity.spell_checker(city=city)
        # если True
        if adv_spell_check:
            # запускаем расширенную проверку опечаток или сокращений
            city = FindCity.advanced_spell_checker(city=city, dataset=self.dataset)
        # поучаем вектор имени города
        full_city_vector = self.model.encode([city], device=self.device)
        # выбираем количество схожих городов для вывода
        tops = min(top_k, len(self.cities_emb))
        # получаем результат при помощи метода util.semantic_search, где по дефолту косинусное сходство
        score = util.semantic_search(full_city_vector, self.cities_emb, top_k=tops)[0]
        # список индексов имен городов из датасета
        lst_idx = [score[i]["corpus_id"] for i in range(len(score))]
        # список с косинусным сходством по индексу
        scores = [score[i]["score"] for i in range(len(score))]
        # формируем результирующий датасет из входного по отобранным индексам
        result_df = self.dataset[self.cols_output].iloc[lst_idx]
        # добавляем колонку со скорингом
        result_df["cos_sim_score"] = scores
        # если нужен вывод в виде словаря
        if output_dict_json:
            # формируем словарь из датафрейма
            output_dict = result_df.to_dict(orient="records")
            # если нужно – то сохраняем json файл
            if save_json_file:
                with open(os.path.join(work_dir, f"{city}.json"), "w") as fp:
                    json.dump(output_dict, fp)
            # возвращаем список словарей
            return output_dict
        # иначе
        else:
            # возвращаем датафрейм
            return result_df


# файл с классом и функциями для работы с датасетами
import os
import pandas as pd
import numpy as np
import gc
import torch
from sentence_transformers import SentenceTransformer

RANDOM = 12345
torch.manual_seed(RANDOM)
np.random.seed(RANDOM)


class DatasetLoader:
    """
    Класс DatasetLoader.
    Класс загружает датасет из txt или csv файлов и создает датасет с векторными
    представлениями слов из созданных датасетов.

    Методы класса:
        load_dataset: метод загрузчик датасета из файла txt или csv,
        load_city_embeddings: метод для создания датасета с векторами слов,
        save_dataset_to_file: сохраняет датасет в файл,
        is_accessible (staticmethod): статический метод для проверки доступности файлов в режиме чтения.
    """

    def __init__(self, work_dir=None):
        """
        Инициализация объекта класса DatasetLoader.
        Параметр:
            work_dir (str): рабочая директория с файлами для датасета, по умолчанию None.
        """
        self.work_dir = work_dir

    @staticmethod
    def is_accessible(file, work_dir, mode="r"):
        """
        Статический метод is_accessible класса DatasetLoader.
        Проверка, является ли файл в рабочей директории
        доступным для работы в предоставленом `mode` формате.
        Параметры:
            file (str): файл с данными для датасета,
            work_dir (str): рабочая директория с файлами для датасета,
            mode (str): режим доступа к файлу, по умолчанию равно 'r' - чтение.
        Возвращаемое значение.
            Boolean: True или False.
        """
        try:
            f = open(os.path.join(work_dir, file), mode)
            f.close()
        except IOError:
            return False
        return True

    def load_dataset(self, file=None, df_cols=None, use_cols=None, col_types=None):
        """
        Метод load_dataset для загрузки датасета из файла txt или csv.
        Параметры:
            file (str): файл с данными для датасета, по умолчанию None,
            df_cols (list): полный список с названиями колонок в датасете, по умолчанию равно None,
            use_cols (list): список с названиями колонок, которые будут отображены в датасете,
                             по умолчанию равно None,
            col_types (dict): словарь с колонками и типами, например {"col_name": str}, для
                              изначального назначения типов данных столбцу или столбцам,
                              по умолчанию равно None.
        Возвращаемое значение:
            dataset (pd. Dataframe): созданный датафрейм Pandas.
        """
        # проверка типа переменной file на строковое значение
        if not isinstance(file, str):
            raise TypeError(
                f"Не соответствует тип переменной file, должен быть тип str. Датасет не будет создан."
            )
        # проверка переменной file на ненулевое значение
        if len(file) == 0:
            raise ValueError(
                f"file не должен быть пустой строкой, списком. Датасет не будет создан."
            )
        # вызов статического метода is_accessible для проверки файла на доступность
        flag = DatasetLoader.is_accessible(file, self.work_dir)
        # если True, то создаем датасет
        if flag:
            print(f"Создаем датасет из файла {file}...")
            # стандарнтый pd.read_csv метод пандас для загрузки датасета
            dataset = pd.read_csv(
                os.path.join(self.work_dir, file),
                header=None,
                names=df_cols,
                usecols=use_cols,
                dtype=col_types,
                delimiter="\t",
                low_memory=False,
            )
            print(f"Датасет создан!")
            # возвращаемый датасет
            return dataset
        #
        else:
            # в случае отсутствия файла возврат ValueError
            raise ValueError(f"Файл {file} на найден в директории {self.work_dir}.")

    def load_city_embeddings(
            self,
            device="cpu",
            model_id=None,
            batch_size=8,
            id_emb_col=None,
    ):
        """
        Метод load_city_embeddings для создания датасета и векторов слов из колонки датасета.
        Параметры:
            device (str): акселератор для создания векторов CPU или GPU, по умолчанию равно 'cpu',
            model_id (str): имя модели для векторизации, по умолчанию равно None,
            batch_size (int): размер батча для создания векторов слов, по умолчанию равно 8,
            id_emb_col (pd.Series или list): столбец с текстом для векторизации, по умолчанию равно None,
            save_to_file (bool): флаг, указывающий, нужно ли сохранять датасет в файл, по умолчанию равно False,
            file_name (str): имя файла для сохранения датасета, по умолчанию равно 'embeddings',
            dir_to_save (str): директория для сохранения датасета.
        Возвращаемое значение:
            dataset (pd.Dataframe): созданный датафрейм Pandas.
        """
        # проверка типа переменной id_emb_col на тип list или pd.Series
        if not isinstance(id_emb_col, (list, pd.Series)):
            raise TypeError(
                f"Не соответствует тип переменной id_emb_col, должен быть тип list или pd.Series. Датасет не будет "
                f"создан."
            )
        # проверка переменной id_emb_col на ненулевое значение
        if len(id_emb_col) == 0:
            raise ValueError(
                f"Количество элементов в id_emb_col не должно быть нулевым. Датасет не будет создан."
            )
        # если проверки пройдены создаём датасет
        else:
            # создаём пустой датасет
            dataset = pd.DataFrame()
            # в список берем только уникальные названия городов
            id_emb_col = list(set(id_emb_col))
            # создаем столбец с названиями городов
            dataset["name"] = id_emb_col
            # загрузка модели для создания векторов
            print(f"Загружаем модель для создания эмбеддингов ...")
            model = SentenceTransformer(model_id)
            # создание векторов
            print(
                f"Создание эмбеддингов...  Размер батча --> {batch_size}, CPU или GPU --> {device} ..."
            )
            embeddings = model.encode(
                id_emb_col, show_progress_bar=True, device=device, batch_size=batch_size
            )
            # добавление в датасет столбца с векторами слов
            dataset["embeddings"] = list(embeddings)
            print(f"Датасет создан!")
            # удаление переменных и очистка памяти CUDA
            del model
            del embeddings
            del id_emb_col
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            # возвращаемый датасет
            return dataset

    def save_dataset_to_file(self, dataset=None, file_name=None, dir_to_save=None):
        """
        Метод save_dataset_to_file для сохранения датасета в файл на диске.
        Параметры:
            dataset (pd.Dataframe): датафрейм Pandas для сохранения на диск.
            file_name (str): имя файла для сохранения датасета, по умолчанию равно'embeddings',
            dir_to_save (str): директория для сохранения датасета.
        Возвращаемое значение:
            Остсутствует.
        """
        # сохраняем датасет в файл методом to_pickle с zip компрессией для оптимизации места на диске
        print(f"Сохраняем датасет в файл {file_name} ...")
        dataset.to_pickle(os.path.join(dir_to_save, file_name), compression="zip")


def reduce_mem_usage(df):
    """
    Функция перебирает все столбцы датафрейма и изменяеет тип данных, чтобы
    уменьшить использование памяти
    Параметр:
            df (pd.Dataframe): исходный датасет.
    Возвращаемое значение:
            df (pd.Dataframe): оптимизированный датасет.
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Память занимаемая датасетом в ОП до обработки: {:.4f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if (
                col_type != object
                and col_type.name != "category"
                and "datetime" not in col_type.name
        ):
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                        c_min > np.finfo(np.float16).min
                        and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                        c_min > np.finfo(np.float32).min
                        and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        elif "datetime" not in col_type.name:
            df[col] = df[col].astype("object")

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Память занимаемая датасетом в ОП после обработки: {:.4f} MB'.format(end_mem))
    print('Экономия {:.2f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


def preprocess_data(dataset=None, city_or_country=None):
    """
    Функция предобработки датасетов с городами или странами.
    Параметры:
            dataset (pd.Dataframe): исходный датасет, по умолчанию равно None,
            city_or_country (str): выбора датасета с городами или странами,
                                   по умолчанию равно None.
                                   Если параметр = "city", то обрабатывается датасет с городами,
                                   если параметр = "country", то обрабатывается датасет с городами,
                                   в противном случае выводистя сообщение о невозможности обработки.
    Возвращаемое значение:
            dataset (pd.Dataframe): обработанный датасет.
    """
    # если датасет с городами
    if city_or_country == "city":
        dataset = dataset.copy()
        # добавляем пробел в столбце alternatenames
        dataset["alternatenames"] = dataset["alternatenames"].str.replace(",", ", ")
        # удаляем строки с попусками в стобцах admin_1_code, name, country_code_iso
        dataset = dataset.dropna(subset=["admin_1_code", "name", "country_code_iso"])
        # создаём новый столбец admin_code в виде суммы через точку столбцов country_code_iso и admin_1_code
        dataset.loc[:, "admin_code"] = (
                dataset["country_code_iso"] + "." + dataset["admin_1_code"]
        )
        # удаляем столбец admin_1_code
        dataset = dataset.drop(["admin_1_code"], axis=1)
        # заполняем пропуски в столбце asciiname на основе столбца names
        dataset.loc[dataset["alternatenames"].isna(), "alternatenames"] = dataset[
            "name"
        ]
        # возврат iso кода NA Намибии
        dataset.loc[dataset["asciiname"].isna(), "asciiname"] = dataset["name"]
        print("Датасет обработан!")
    # если датасет со странами
    elif city_or_country == "country":
        dataset = dataset.copy()
        # добавляем пробел в столбце languages
        dataset["languages"] = dataset["languages"].str.replace(",", ", ")
        #
        dataset.loc[dataset["country"] == "Namibia", "iso"] = "NA"
        print("Датасет обработан!")
    # в противном случае
    else:
        print(
            """
              Не выбран датасет для обработки.
              Вы вабрали значение параметра city_or_country={}.
              Нужно указать правильное значение параметра city_or_country:
              - для городов - 'city';
              - для стран - 'country'.
              Параметр д.б. строкой.
              """.format(
                city_or_country
            )
        )
    # возврат результата обработки
    return dataset


def remove_difference(cities=None, admin_codes=None):
    """
    Функция для устранения разницы в значениях между полями:
     - admin_code в датасете с городами,
     — admin_code в датасете с областями.
    Т.к. в таблице с городами admin_code это внешний ключ, ссылающийся
    на первичный ключ admin_code в таблице с областями, значения ключа, которые
    есть в таблице с городами, должны обязательно быть в таблице с областями.
    Параметры:
            cities (pd.Dataframe): датасет с городами, по умолчанию равно None,
            admin_codes (pd.Dataframe): датасет с областями, по умолчанию равно None
    Возвращаемое значение:
            admin_codes (pd.Dataframe): обработанный датасет с областями.
    """
    # делаем список из значений, которые есть в cities, но нет в admin_codes
    # для этого находим разность множеств cities["admin_code"] и admin_codes["admin_code"]
    difference = list(set(cities["admin_code"]) - set(admin_codes["admin_code"]))
    # создаём новый датафрейм с один столбцом admin_code и значениями из списка difference
    new_rows = pd.DataFrame({"admin_code": difference})
    # добавление значений из списка в столбец DataFrame
    admin_codes = pd.concat([admin_codes, new_rows], ignore_index=True)
    # заполняем пропуски в столбцах "name", "name_ascii" значение No admin, т.к неизвестно к какой области это
    # относится.
    admin_codes.loc[
        admin_codes["admin_code"].isin(difference), ["name", "name_ascii"]
    ] = "No admin"
    return admin_codes

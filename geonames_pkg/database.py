# файл с классами для работы с БД
# базовые импорты
import pandas as pd
# импорты для работы с БД
from sqlalchemy import (
    create_engine,
    text,
)
from psycopg2.extensions import AsIs


class CreateDatabase:
    """
    Класс CreateDatabase для создания базы данных.
    """

    def __init__(self, conn_str=None):
        """
        Инициализация объекта класса CreateDatabase.

        Параметры:
            conn_str (str): строка подключения к базе данных,
                            по умолчанию равно None.
        """
        self.conn_str = conn_str

    def create_db(self, db_name=None):
        """
        Метод create_db.
        Создание базы данных.

        Параметры:
            db_name (str, optional): имя создаваемой базы данных,
                                     по умолчанию равно None.
        """
        # создание подключения к базе данных
        engine = create_engine(self.conn_str)
        conn = engine.connect()

        # выполнение SQL-запроса для создания базы данных
        conn.execution_options(isolation_level="AUTOCOMMIT").execute(
            text(f"CREATE DATABASE {db_name}")
        )
        conn.close()  # Закрытие соединения

        # вывод сообщения об успешном создании базы данных
        print(f"База данных {db_name} создана!")


class DataFrameSQL:
    """
    Класс для сохранения данных из Pandas DataFrame в базу данных Postgres и наоборот.
    """

    def __init__(self, engine):
        """
        Инициализация объекта для работы с базой данных.

        Параметры:
            engine (sqlalchemy.engine): объект SQLAlchemy Engine для подключения к базе данных.
        """
        self.engine = engine

    def to_sql(
            self,
            df,
            table_name,
            if_exists="append",
            chunksize=10000,
            method="multi",
            index=False,
            dtype=None,
            fk_restriction=False,
    ):
        """
        Метод to_sql.
        Сохраняет DataFrame в базу данных.

         Параметры:
               df (pd.DataFrame): датафрейм, который нужно сохранить,
               table_name (str): наименование таблицы в базе данных, в которую нужно сохранить DataFrame,
               if_exists (bool): опция для действий при конфликте существующих записей, по умолчанию
                                 равно 'append',
               chunksize (int): количество строк для записи за один запрос к базе данных, по умолчанию
                                равно 10000,
               method (str): метод вставки данных в базу данных, по умолчанию равно 'multi',
               index (boll): опция для включения индекса в базу данных, по умолчанию равно False,
               dtype (dict): словарь для указания типов данных столбцов при сохранении в базу данных,
                             по умолчанию равно None,
               fk_restriction (bool): опция для управления ограничениями внешнего ключа при сохранении
                                      данных, по умолчанию равно False.
        """
        # если True
        if fk_restriction:
            # снятие ограничения внешнего ключа в зависимости от имени таблицы
            with self.engine.connect() as conn:
                if table_name == "embeddings":
                    conn.execute(text("ALTER TABLE city DROP CONSTRAINT fk_name"))
                    conn.commit()
                elif table_name == "country":
                    conn.execute(
                        text("ALTER TABLE city DROP CONSTRAINT fk_country_code_iso")
                    )
                    conn.commit()
                else:
                    conn.execute(text("ALTER TABLE city DROP CONSTRAINT fk_admin_code"))
                    conn.commit()
            conn.close()

        # загрузка данных из DataFrame в базу данных
        print(f"Загружаем датафрейм в таблицу {table_name} базы данных geonames ...")
        df.to_sql(
            table_name,
            con=self.engine,
            if_exists=if_exists,
            chunksize=chunksize,
            method=method,
            index=index,
            dtype=dtype,
        )
        print(f"Загружено {len(df)} записей!")
        # если True
        if fk_restriction:
            # восстановление первичного ключа и ограничений внешнего ключа после загрузки данных
            with self.engine.connect() as conn:
                if table_name == "embeddings":
                    conn.execute(text("ALTER TABLE embeddings ADD PRIMARY KEY (name)"))
                    conn.commit()
                    conn.execute(
                        text(
                            "ALTER TABLE city ADD CONSTRAINT fk_name FOREIGN KEY (name) REFERENCES embeddings(name)"
                        )
                    )
                    conn.commit()
                elif table_name == "country":
                    conn.execute(text("ALTER TABLE iso ADD PRIMARY KEY (country)"))
                    conn.commit()
                    conn.execute(
                        text(
                            "ALTER TABLE city ADD CONSTRAINT fk_country_code_iso FOREIGN KEY (country_code_iso) "
                            "REFERENCES country(iso)"
                        )
                    )
                    conn.commit()
                else:
                    conn.execute(
                        text("ALTER TABLE admincode ADD PRIMARY KEY (admin_code)")
                    )
                    conn.commit()
                    conn.execute(
                        text(
                            "ALTER TABLE city ADD CONSTRAINT fk_admin_code FOREIGN KEY (admin_code) REFERENCES "
                            "admincode(admin_code)"
                        )
                    )
                    conn.commit()
            conn.close()

    @staticmethod
    def check_country(countries):
        """
        Статический метод check_country класса DataFrameSQL.
        Метод проверяет тип введенной перемменой countries и
        преобразовывает значение переменной в строку для формирования запроса к БД.
        Например, сначала список ["Russia", "Kazakhstan"] преобразовывается в вид
        ["'Russia'", "'Kazakhstan'"], т.е. добавляются кавычки и затем список преобразовывается в строку.
        Это необходимо для передачи в конструкцию WHERE ... IN () ... SQL запроса.
         Параметры:
               countries(str): страна или список стран.
         Возвращаемое значение:
               countries (str): переработанная переменная для запроса query.
         """
        # проверка на соответствие переменной countries на тип str или list
        if not isinstance(countries, (str, list)):
            raise TypeError(
                f"Не соответствует тип переменной countries, должен быть тип str или list. Датасет не будет создан."
            )
        # проверка переменной countries на ненулевую длину
        if len(countries) == 0:
            raise ValueError(
                f"Пустая строка или пустой список вместо countries. . Датасет не будет создан."
            )
        # если на вход подается строка с одной страной, то добавляются кавычки и возвращается измененная строка
        if isinstance(countries, str):
            countries = "'" + countries + "'"
            return countries
        # если на вход подается список с одной или несколькими странами, то добавляются кавычки к каждому элементу и
        # возвращается измененная строка
        else:
            countries = ["'" + country + "'" for country in countries]
            return ", ".join(countries)

    def from_sql(self, query=None, countries=None, population=15000):
        """
        Метод from_sql.
        Загружает данные из БД Postgres в DataFrame Pandas.

         Параметры:
               query (str): SQL запрос к БД, по умолчанию равно None,
               countries (str or list): страна или список стран для ограничения в запросе по странам,
                                        по умолчанию равно None,
               population (int): население в городах, по умолчанию равно 15000.
         """
        # преобразование стран через вызов статического метода check_country
        countries = DataFrameSQL.check_country(countries)
        # запрос к БД
        # т.к. в запросе есть условие WHERE co.country IN ({}) AND ci.population >= {}
        # необходимо добавить format(countries, population), чтобы страны и население отобразились в запросе
        query = query.format(
            countries, population
        )
        # метод пандас для получения датасета из БД
        dataset = pd.read_sql(query, con=self.engine)
        # возвращаемый датасет
        return dataset


def addapt_numpy_float32(numpy_float32):
    """
    Функция адаптер типа np.float32.
     Параметры:
           принимает объект типа numpy.float32
     Возвращаемое значение:
           numpy.float32 через  класс-обертку AsIs
    """
    return AsIs(numpy_float32)

# файл для конфигурации переменных
import os
import torch
# Переменные для датасетов
# рабочая директория проекта
WORK_DIR = os.path.abspath(os.curdir)
# директория c txt исходниками проекта
SRC_DIR = os.path.join(WORK_DIR, 'source')
# директория для сохранения датасетов
DATA_DIR = os.path.join(WORK_DIR, 'datasets')
# директория для сохранения json файлов с результатом
OUT_DIR = os.path.join(WORK_DIR, 'output')

# Нижеперечисленные переменные будут использованы при загрузке в методе load_dataset класса DatasetLoader
CITY_FILE = "cities500.txt"  # файл с городами
COUNTRY_FILE = "countryInfo.txt"  # файл со странами
ADMIN_CODE_FILE = "admin1CodesASCII.txt"  # файл с областями
# названия столбцов для датасета с городами
CITY_COLS = [
    "city_geoname_id",
    "name",
    "asciiname",
    "alternatenames",
    "latitude",
    "longitude",
    "feature_class",
    "feature_code",
    "country_code_iso",
    "cc2",
    "admin_1_code",
    "admin_2_code",
    "admin_3_code",
    "admin_4_code",
    "population",
    "elevation",
    "dem",
    "timezone",
    "modification_date",
]
# названия столбцов которые будут выведены в датасет с городами
USE_CITY_COLS = [
    "city_geoname_id",
    "name",
    "asciiname",
    "alternatenames",
    "latitude",
    "longitude",
    "feature_class",
    "feature_code",
    "country_code_iso",
    "admin_1_code",
    "population",
    "timezone",
]
# типы данных для некоторых столбцов, заданные по умолчанию при загрузке
COL_TYPES = {"country_code_iso": str, "admin_1_code": str}
# названия столбцов для датасета со странами
COUNTRY_COLS = [
    "iso",
    "iso_3",
    "iso_numeric",
    "fips",
    "country",
    "capital",
    "area_in_sq_km",
    "population",
    "continent",
    "tld",
    "currency_code",
    "currency_name",
    "phone",
    "postal_code_format",
    "postal_code_regex",
    "languages",
    "county_geoname_id",
    "neighbours",
    "equivalent_fips_code",
]
# названия столбцов которые будут выведены в датасет со странами
USE_COUNTRY_COLS = [
    "iso",
    "iso_3",
    "country",
    "capital",
    "area_in_sq_km",
    "population",
    "continent",
    "tld",
    "currency_code",
    "currency_name",
    "phone",
    "languages",
]
# названия столбцов для датасета с областями
ADMIN_COLS = ["admin_code", "name", "name_ascii", "geoname_id"]
# названия столбцов которые будут выведены в датасет с областями
USE_ADMIN_COLS = ["admin_code", "name", "name_ascii"]

# Переменные для БД
# словарь с конфигурацией для подключения к БД
db_config = {
    "user": "andreivk",  # имя пользователя
    "pwd": "andreivk_1980",  # пароль
    "host": "localhost",  # хост
    "port": 5432,  # порт подключения
    "db": "geonames",  # название базы данных
    "default_db": "postgres",  # дефолтная база данных для создания новой БД
}
# коннекшн строка для инициализации подключения к БД geonames
CONN_STR_GEONAMES = "postgresql://{}:{}@{}:{}/{}".format(
    db_config["user"],
    db_config["pwd"],
    db_config["host"],
    db_config["port"],
    db_config["db"],
)
# коннекшн строка для инициализации подключения к дефолтной БД postgres для создания БД geonames
CONN_STR_DEFAULT = "postgresql://{}:{}@{}:{}/{}".format(
    db_config["user"],
    db_config["pwd"],
    db_config["host"],
    db_config["port"],
    db_config["default_db"],
)

# список стран для поиска
# по умолчанию взяты Россия и Казахстан,
# но можно составить список с изначальным условием, правильное написание стран в файле countryInfo.txt
# ["Russia", "Kazakhstan", "Belarus", "Armenia", "Azerbaijan", "Georgia", "Serbia", Turkey"]
COUNTRIES_LST = ["Russia", "Kazakhstan"]
# численность населения
POPULATION = 15000
# SQL - запрос к БД
QUERY = """
  SELECT ci.city_geoname_id as geoname_id,
        ci.name,
        ci.alternatenames,
        ad.name as oblast,
        co.country,
        co.capital,
        co.currency_name,
        ci.timezone,
        ci.latitude,
        ci.longitude,
        em.embeddings
  FROM city AS ci
  JOIN country AS co ON ci.country_code_iso = co.iso
  JOIN embeddings AS em ON em.name = ci.name
  JOIN admincode AS ad ON ad.admin_code = ci.admin_code 
  WHERE co.country IN ({}) AND ci.population >= {}
  ORDER BY ci.name ASC;
         """

# Переменные для моделирования эмбеддингов и вывода результата
# проверяем наличие gpu
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
# имя модели sentence-transformers
MODEL_ID = "sentence-transformers/LaBSE"
# список выводимых столбцов для результирующей таблицы.
COLS_OUTPUT = [
    "geoname_id",
    "name",
    "oblast",
    "country",
    "capital",
    "currency_name",
    "timezone",
    "latitude",
    "longitude",
]

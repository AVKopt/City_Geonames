# скрипт для заполнения таблиц базы данных
from config import CONN_STR_GEONAMES, DATA_DIR
from database import DataFrameSQL, addapt_numpy_float32
import os
import gc
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, ARRAY, REAL
from psycopg2.extensions import register_adapter


def main():
    # создаем словарь, в котором будем хранить загруженные DataFrame
    dataframes = {}
    # загружаем DataFrame и сохраняем его в словаре
    for file_name in ["cities", "countries", "admin_codes", "embeddings"]:
        dataset = pd.read_pickle(os.path.join(DATA_DIR, file_name), compression="zip")
        dataframes[file_name] = dataset
    # датафрейм с городами
    cities = dataframes["cities"]
    # датафрейм со странами
    countries = dataframes["countries"]
    # датафрейм с областями
    admin_codes = dataframes["admin_codes"]
    # датафрейм с веторами
    embeddings = dataframes["embeddings"]
    # применение register_adapter
    register_adapter(np.float32, addapt_numpy_float32)
    # создание подключения к БД
    engine = create_engine(CONN_STR_GEONAMES)
    # создаём экземпляра класса DataFrameSQL
    data_sql = DataFrameSQL(engine)
    # cохраняем данные в таблицу 'admincode'
    data_sql.to_sql(admin_codes, "admincode")
    # cохраняем данные в таблицу 'embeddings' с использованием параметра dtype для столбца 'embeddings'
    data_sql.to_sql(embeddings, "embeddings", dtype={"embeddings": ARRAY(REAL)})
    # cохраняем данные в таблицу 'country'
    data_sql.to_sql(countries, "country")
    # cохраняем данные в таблицу 'city'
    data_sql.to_sql(cities, "city")
    # очистка памяти
    del cities, countries, admin_codes, embeddings
    gc.collect()


if __name__ == "__main__":
    main()

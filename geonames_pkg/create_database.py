# скрипт для создания базы данных
from config import CONN_STR_DEFAULT, CONN_STR_GEONAMES
from database import CreateDatabase
from tables import Base
from sqlalchemy import create_engine


def main():
    # создаем объект класса CreateDatabase
    database = CreateDatabase(conn_str=CONN_STR_DEFAULT)
    # методом create_db создаем БД geonames
    database.create_db(db_name="geonames")
    # подключение к созданной БД
    engine = create_engine(CONN_STR_GEONAMES)
    # создание таблиц через declarative_base()
    Base.metadata.create_all(engine)


if __name__ == "__main__":
    main()

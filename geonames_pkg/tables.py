# импорты для работы с БД
from sqlalchemy import (
    ARRAY,
    REAL,
    Column,
    Float,
    ForeignKey,
    Integer,
    String,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Vectors(Base):
    """
    Класс Vectors для хранения векторных представлений географических названий.
    """
    # имя таблицы
    __tablename__ = "embeddings"
    # комментарий с описанием таблицы
    __table_args__ = {
        "comment": "Таблица с векторными представлениями географических названий."
    }
    # задаем в переменные параметры столбцов в таблице БД, имя переменной является именем столбца
    name = Column(
        String,
        nullable=False,
        unique=True,
        primary_key=True,
        comment="Наименование географического объекта"
    )
    embeddings = Column(
        ARRAY(REAL),
        comment="Векторные представления географического объекта"
    )

    def __repr__(self):
        """
        Метод __repr__.
        Возвращает строковое представление объекта.

        Возвращаемое значение:
            Строка (str): строковое представление объекта с названиями столбцов в таблице.
                          Столбцы:
                            - name,
                            - embeddings.
        """
        return f"{self.name} {self.embeddings}"


class Country(Base):
    """
    Класс Country для хранения информации о странах.
    """
    # имя таблицы
    __tablename__ = "country"
    # комментарий с описанием таблицы
    __tableargs__ = {"comment": "Таблица со странами."}
    # задаем в переменные параметры столбцов в таблице БД, имя переменной является именем столбца
    iso = Column(
        String,
        nullable=False,
        unique=True,
        primary_key=True,
        comment="id, ISO 2-letter country code",
    )
    iso_3 = Column(String, comment="ISO 3-letter country code")
    country = Column(String, comment="country name")
    capital = Column(String, comment="country capital")
    area_in_sq_km = Column(Float, comment="country area")
    population = Column(String, comment="country population")
    continent = Column(String, comment="country continent")
    tld = Column(String, comment="country domen")
    currency_code = Column(String, comment="country currency code")
    currency_name = Column(String, comment="country currency name")
    phone = Column(String, comment="country phone code")
    languages = Column(String, comment="country base languages")

    def __repr__(self):
        """
        Метод __repr__.
        Возвращает строковое представление объекта.

        Возвращаемое значение:
            Строка (str): строковое представление объекта с названиями столбцов в таблице.
        """
        return f"{self.iso} {self.iso_3} {self.country} {self.capital} {self.area_in_sq_km} {self.population} {self.continent} {self.tld} {self.currency_code} {self.currency_name} {self.phone} {self.languages}"


class Admin(Base):
    """
    Класс Admin для хранения информации об областях.
    """
    # имя таблицы
    __tablename__ = "admincode"
    # комментарий с описанием таблицы
    __tableargs__ = {"comment": "Таблица с областями."}

    # задаем в переменные параметры столбцов в таблице БД, имя переменной является именем столбца
    admin_code = Column(String, nullable=False, unique=True, primary_key=True)
    name = Column(String, comment="name of geographical point (utf8)")
    name_ascii = Column(
        String, comment="name of geographical point in plain ascii characters"
    )

    def __repr__(self):
        """
        Метод __repr__.
        Возвращает строковое представление объекта.

        Возвращаемое значение:
            Строка (str): строковое представление объекта с названиями столбцов в таблице.
        """
        return f"{self.admin_code} {self.name} {self.name_ascii}"


class City(Base):
    """
    Класс City для хранения информации о городах.
    """
    # имя таблицы
    __tablename__ = "city"
    # комментарий с описанием таблицы
    __tableargs__ = {"comment": "Таблица с городами."}
    # задаем в переменные параметры столбцов в таблице БД, имя переменной является именем столбца
    city_geoname_id = Column(
        Integer,
        nullable=False,
        unique=True,
        primary_key=True,
        autoincrement=True,
        comment="id of record in geonames database",
    )
    name = Column(
        String,
        ForeignKey(
            "embeddings.name",
            name="fk_name",
            onupdate="CASCADE",
            ondelete="CASCADE",
        ),
        comment="name of geographical point (utf8)",
    )
    asciiname = Column(
        String, comment="name of geographical point in plain ascii characters"
    )
    alternatenames = Column(
        String,
        comment="alternatenames, comma separated, ascii names automatically transliterated, convenience attribute from alternatename table",
    )
    latitude = Column(Float, comment="latitude in decimal degrees")
    longitude = Column(Float, comment="longitude in decimal degrees")
    feature_class = Column(
        String, comment="see http://www.geonames.org/export/codes.html"
    )
    feature_code = Column(
        String, comment="see http://www.geonames.org/export/codes.html"
    )
    country_code_iso = Column(
        String,
        ForeignKey(
            "country.iso",
            name="fk_country_code_iso",
            onupdate="CASCADE",
            ondelete="CASCADE",
        ),
        comment="ISO-3166 2-letter country code, FK",
    )
    population = Column(Integer, comment="population")
    timezone = Column(String, comment="the iana timezone")
    admin_code = Column(
        String,
        ForeignKey(
            "admincode.admin_code",
            name="fk_admin_code",
            onupdate="CASCADE",
            ondelete="CASCADE",
        ),
        comment="code of admin division",
    )
    # задаем взаимосвязи между таблицами
    country = relationship("Country", backref="quote_country", lazy="subquery")
    admincode = relationship("Admin", backref="quote_admincode", lazy="subquery")
    embeddings = relationship("Vectors", backref="quote_embeddings", lazy="subquery")

    def __repr__(self):
        """
        Метод __repr__.
        Возвращает строковое представление объекта.

        Возвращаемое значение:
            Строка (str): строковое представление объекта с названиями столбцов в таблице.
        """
        return f"{self.city_geoname_id} {self.name} {self.asciiname} {self.alternatenames} {self.latitude} {self.longitude} {self.feature_class} {self.feature_code} {self.country_code_iso} {self.population} {self.timezone} {self.admin_code}"

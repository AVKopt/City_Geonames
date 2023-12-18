# скрипт для создания и сохранения датасетов
from config import (
    SRC_DIR,
    DATA_DIR,
    DEVICE,
    CITY_FILE,
    CITY_COLS,
    USE_CITY_COLS,
    COL_TYPES,
    COUNTRY_FILE,
    COUNTRY_COLS,
    USE_COUNTRY_COLS,
    ADMIN_CODE_FILE,
    ADMIN_COLS,
    USE_ADMIN_COLS,
    MODEL_ID
)
from dataset import DatasetLoader, reduce_mem_usage, remove_difference, preprocess_data
import gc


def main():
    # создаем объект loader класса DatasetLoader
    loader = DatasetLoader(work_dir=SRC_DIR)
    # создаем датафрейм с городами, пропуская через функцию reduce_mem_usage
    cities = reduce_mem_usage(
        loader.load_dataset(
            file=CITY_FILE, df_cols=CITY_COLS, use_cols=USE_CITY_COLS, col_types=COL_TYPES
        )
    )
    # преодбработка датафрейма с городами
    cities = preprocess_data(dataset=cities, city_or_country="city")
    # создаем датафрейм со странами, пропуская через функцию reduce_mem_usage
    countries = reduce_mem_usage(
        loader.load_dataset(
            file=COUNTRY_FILE, df_cols=COUNTRY_COLS, use_cols=USE_COUNTRY_COLS
        )
    )
    # преодбработка датафрейма со странами
    countries = preprocess_data(dataset=countries, city_or_country="country")
    # создаем датафрейм с областями, пропуская через функцию reduce_mem_usage
    admin_codes = reduce_mem_usage(
        loader.load_dataset(
            file=ADMIN_CODE_FILE, df_cols=ADMIN_COLS, use_cols=USE_ADMIN_COLS
        )
    )
    # преодбработка датафрейма с областями
    admin_codes = remove_difference(cities=cities, admin_codes=admin_codes)
    # создаем датафрейм с векторами имен городов
    embeddings = loader.load_city_embeddings(
        device=DEVICE, model_id=MODEL_ID, id_emb_col=cities["name"]
    )
    # сохраняем датафреймы на диск с zip компрессией
    for dataset, file_name in zip([cities, countries, admin_codes, embeddings],
                                  ["cities", "countries", "admin_codes", "embeddings"]):
        loader.save_dataset_to_file(dataset=dataset, file_name=file_name, dir_to_save=DATA_DIR)
    # очистка памяти
    del cities, countries, admin_codes, embeddings
    gc.collect()
    print("Создание и сохранение датасетов закончено!")


if __name__ == "__main__":
    main()

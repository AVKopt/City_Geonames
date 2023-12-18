# главный исполняемый скрипт проекта
from flask import Flask, render_template, request
from config import CONN_STR_GEONAMES, QUERY, COUNTRIES_LST, POPULATION, COLS_OUTPUT, MODEL_ID, OUT_DIR
from finder import FindCity
from database import DataFrameSQL
from sqlalchemy import create_engine

app = Flask(__name__)


# функция получения данных из БД
def get_data():
    # создаем подключение
    engine = create_engine(CONN_STR_GEONAMES)
    # инициализируем объект класса DataFrameSQL
    data_loader = DataFrameSQL(engine=engine)
    # формируем датасет с данными согласно запросу, списку стран и населению из config файла
    df = data_loader.from_sql(query=QUERY, countries=COUNTRIES_LST, population=POPULATION)
    # возврат датафрейма
    return df


# вызов функции get_data()
data = get_data()
# инициализируем объект класса FindCity с параметрами из config файла
finder = FindCity(model_id=MODEL_ID, device="cpu", dataset=data,
                  emb_col="embeddings", cols_output=COLS_OUTPUT)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        # получаем город из файла index.html
        city = request.form['city']
        # получаем кол-во городов для вывода из файла index.html
        top_k = int(request.form['top_k'])
        # получаем флаг расширенной проверки для вывода из файла index.html
        adv_spell_check = bool(request.form.get('adv_spell_check'))
        # получаем флаг нужен ли вывод в словарь из файла index.html
        output_dict_json = bool(request.form.get('output_dict_json'))
        # методом get_city класса FindCity получаем результат
        result = finder.get_city(city=city, top_k=top_k, adv_spell_check=adv_spell_check,
                                 output_dict_json=output_dict_json, work_dir=OUT_DIR)
        if isinstance(result, list) and all(isinstance(d, dict) for d in result):
            # если результат - список словарей, подготовим его для отображения в шаблоне
            return render_template('index.html', result_list=result)
        else:
            # если результат не является списком словарей, предполагаем, что это DataFrame
            result_html = result.to_html(classes='data', header="true")
            return render_template('index.html', tables=[result_html], titles=result.columns.values)


if __name__ == '__main__':
    app.run(debug=True)

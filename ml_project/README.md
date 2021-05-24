Heart prediction
==============================

## Project structure
~~~
.
├── configs                         <- configurations for training and logging
│   ├── logging_config.yml
│   ├── train_config_knn.yml
│   ├── train_config_logreg.yml
│   └── train_config_trees.yml
├── ml_project                      <- main project code 
│   ├── data                        <- functionality for data processing
│   │   ├── download_data.sh
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   ├── entities                    <- dataclasses for parameters to operate with configs
│   │   ├── __init__.py
│   │   ├── params.py
│   │   └── train_pipeline_params.py
│   ├── experiments.py              <- ml experiments to search best model parameters
│   ├── features                    <- functionality for features generation
│   │   ├── build_features.py
│   │   └── __init__.py
│   ├── models                      <- functionality for model training and prediction
│   │   ├── __init__.py
│   │   └── model_fit_predict.py
│   └── pipeline.py                 <- console interface to ml pipeline
├── models                          <- saved models in pickle format and metrics 
│   ├── metrics_logreg.json
│   └── model_logreg.pkl
├── notebooks                       <- ipython notebooks 
│   └── EDA.ipynb
├── README.md
├── requirements.txt
├── setup.py
└── tests                           <- tests for separated functionalities and pipeline generally
    ├── configs
    │   └── train_config.yml
    ├── data
    │   └── test_make_data.py
    ├── features
    │   └── test_features.py
    ├── global_fixtures.py
    ├── model
    │   └── test_train_models.py
    └── test_end2end_training.py
~~~

## How to use
Installation: 
~~~
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
pip install .
~~~
Download data:
~~~
bash ./ml_project/data/download_data.sh
~~~
Training:
~~~
python3 ml_project/pipeline.py train --config configs/train_config_logreg.yml
~~~
Prediction (dataset must contain all features specified in config):
~~~
python3 ml_project/pipeline.py predict --dataset data/raw/heart.csv --config configs/train_config_logreg.yml --output data/prediciton.csv
~~~
Help:
~~~
python3 ml_project/pipeline.py --help
~~~

## Tests
~~~
pytest tests/
~~~

## Data
- [heart-disease-uci](https://www.kaggle.com/ronitf/heart-disease-uci?select=heart.csv)

--------------------

## Разбалловка

№ | Описание | Баллы | Выполнено
--- | --- | --- | ---
-2 | Назовите ветку homework1 | 1 балл | ✔️
-1 | положите код в папку ml_project | | ✔️
0 | В описании к пулл реквесту описаны основные "архитектурные" и тактические решения, которые сделаны в вашей работе. В общем, описание что именно вы сделали и для чего, чтобы вашим ревьюерам было легче понять ваш код. | 2 балла | ✔️
1 | Выполнение EDA, закоммитьте ноутбук в папку с ноутбуками | 2 балла | ✔️
2 | Проект имеет модульную структуру(не все в одном файле =) ) | 2 балла | ✔️
3 | использованы логгеры | 2 балла | ✔️
4 | написаны тесты на отдельные модули и на прогон всего пайплайна | 3 балла | ✔️
5 | Для тестов генерируются синтетические данные, приближенные к реальным: 1) можно посмотреть на библиотеки https://faker.readthedocs.io/en/, https://feature-forge.readthedocs.io/en/latest/ ; 2) можно просто руками посоздавать данных, собственноручно написанными функциями; 3) как альтернатива, можно закоммитить файл с подмножеством трейна(это не оценивается)  | 3 балла | ✔️
6 | Обучение модели конфигурируется с помощью конфигов в json или yaml, закоммитьте как минимум 2 корректные конфигурации, с помощью которых можно обучить модель (разные модели, стратегии split, preprocessing) | 3 балла | ✔️
7 | Используются датаклассы для сущностей из конфига, а не голые dict | 3 балла | ✔️
8 | Используйте кастомный трансформер(написанный своими руками) и протестируйте его | 3 балла | ✔️
9 | Обучите модель, запишите в readme как это предлагается | 3 балла | ✔️
10 | напишите функцию predict, которая примет на вход артефакт/ы от обучения, тестовую выборку(без меток) и запишет предикт, напишите в readme как это сделать | 3 балла | ✔️
11 | Используется hydra  (https://hydra.cc/docs/intro/) | 3 балла - доп баллы | 
12 | Настроен CI(прогон тестов, линтера) на основе github actions (будем проходить дальше в курсе, но если есть желание поразбираться - welcome) | 3 балла - доп баллы | 
13 | Проведите самооценку, опишите, в какое колво баллов по вашему мнению стоит оценить вашу работу и почему | 1 балл - доп баллы | ✔️

## Итоговое количество баллов
30 + 1 (за самооценку)
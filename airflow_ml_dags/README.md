Homework 3
-------------------

## Запуск airflow
```
docker-compose up --build
```

## Тестирование кода дагов
```
docker exec -it airflow_ml_dags_scheduler_1 pytest -v .
```

## Выполненная работа

0) Поднял локально airflow
1) Сделал DAG, который генерирует данные каждый день и пишет их в нужное место `data/raw/` (5 баллов)
2) 
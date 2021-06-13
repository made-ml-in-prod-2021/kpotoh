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
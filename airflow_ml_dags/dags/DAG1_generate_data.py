from datetime import timedelta

from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago

DEFAULT_VOLUME = '/home/mr/MADE/DS-22/ml_in_prod/airflow_ml_dags/data:/data'

default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "DAG1_generate_data",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=days_ago(7)
) as dag:
    start_task = DummyOperator(task_id='start-generation')
    generate_data = DockerOperator(
        task_id="airflow-generate",
        image="airflow-generate",
        command="/data/raw/{{ ds }}",
        network_mode="bridge",
        do_xcom_push=False,
        volumes=[DEFAULT_VOLUME]
    )

    end_task = DummyOperator(task_id='end-generation')

    start_task >> generate_data >> end_task

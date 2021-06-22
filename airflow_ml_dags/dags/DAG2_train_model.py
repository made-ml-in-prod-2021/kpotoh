from datetime import timedelta

from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from airflow.models import Variable
from airflow.sensors.filesystem import FileSensor

from utils import DEFAULT_VOLUME, default_args


with DAG(
    "DAG2_train_model",
    default_args=default_args,
    schedule_interval="@weekly",
    start_date=days_ago(7)
) as dag:

    start_task = DummyOperator(task_id='start-train-pipeline')

    data_await = FileSensor(
        task_id="await-data",
        filepath='/opt/airflow/data/raw/{{ ds }}/data.csv',
        poke_interval=10,
        retries=100,
    )
    target_await = FileSensor(
        task_id="await-target",
        filepath='/opt/airflow/data/raw/{{ ds }}/target.csv',
        poke_interval=10,
        retries=100,
    )
    preprocessing = DockerOperator(
        task_id="preprocessing",
        image="airflow-preprocess",
        command="--input-dir data/raw/{{ ds }} "
                "--output-dir data/processed/{{ ds }}",
        network_mode="bridge",
        do_xcom_push=False,
        volumes=[DEFAULT_VOLUME]
    )
    splitting = DockerOperator(
        task_id="splitting",
        image="airflow-split",
        command="--input-dir data/processed/{{ ds }} "
                "--output-dir data/splitted/{{ ds }}",
        network_mode="bridge",
        do_xcom_push=False,
        volumes=[DEFAULT_VOLUME]
    )
    training = DockerOperator(
        task_id="training",
        image="airflow-train",
        command="--data-dir data/splitted/{{ ds }} "
                "--model-dir data/models/{{ ds }}",
        network_mode="bridge",
        do_xcom_push=False,
        volumes=[DEFAULT_VOLUME]
    )      
    validation = DockerOperator(
        task_id="validation",
        image="airflow-validate",
        command="--data-dir data/splitted/{{ ds }} "
                "--model-dir data/models/{{ ds }}",
        network_mode="bridge",
        do_xcom_push=False,
        volumes=[DEFAULT_VOLUME]
    )
    end_task = DummyOperator(task_id='end-train-pipeline')

    start_task >> [data_await, target_await] >> preprocessing >> splitting >> training >> validation >> end_task

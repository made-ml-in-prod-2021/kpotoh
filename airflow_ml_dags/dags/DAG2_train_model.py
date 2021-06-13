from datetime import timedelta

from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago

from utils import DEFAULT_VOLUME, default_args


with DAG(
    "DAG2_train_model",
    default_args=default_args,
    schedule_interval="@weekly",
    start_date=days_ago(7)
) as dag:

    start_task = DummyOperator(task_id='start-train-pipeline')

    preprocessing = DockerOperator(
        task_id="preprocessing",
        image="airflow-preprocess",
        command="--input-dir /data/raw/{{ ds }} "
                "--output-dir /data/processed/{{ ds }}",
        network_mode="bridge",
        do_xcom_push=False,
        volumes=[DEFAULT_VOLUME]
    )
    splitting = DockerOperator(
        task_id="splitting",
        image="airflow-split",
        command="--input-dir /data/processed/{{ ds }} "
                "--output-dir /data/splitted/{{ ds }}",
        network_mode="bridge",
        do_xcom_push=False,
        volumes=[DEFAULT_VOLUME]
    )
    training = DockerOperator(
        task_id="training",
        image="airflow-train",
        command="--data-dir /data/splitted/{{ ds }} "
                "--model-dir /data/models/{{ ds }}",
        network_mode="bridge",
        do_xcom_push=False,
        volumes=[DEFAULT_VOLUME]
    )      
    validating = DockerOperator(
        task_id="validating",
        image="airflow-validate",
        command="--data-dir /data/splitted/{{ ds }} "
                "--model-dir /data/models/{{ ds }}",
        network_mode="bridge",
        do_xcom_push=False,
        volumes=[DEFAULT_VOLUME]
    )
    end_task = DummyOperator(task_id='end-train-pipeline')

    start_task >> preprocessing >> splitting >> training >> validating >> end_task

from datetime import timedelta

from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago

from utils import DEFAULT_VOLUME, default_args


with DAG(
    "DAG3_inference",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=days_ago(3)
) as dag:

    start_task = DummyOperator(task_id='start-prediction')

    preprocessing = DockerOperator(
        task_id="preprocessing",
        image="airflow-preprocess",
        command="--input-dir /data/raw/{{ ds }} "
                "--output-dir /data/processed/for_preds/{{ ds }} "
                "--prediction",
        network_mode="bridge",
        do_xcom_push=False,
        volumes=[DEFAULT_VOLUME]
    )  
    prediction = DockerOperator(
        task_id="prediction",
        image="airflow-predict",
        command="--data-dir /data/processed/for_preds/{{ ds }} "
                "--output-dir /data/predictions/{{ ds }} "
                "--model-dir {{ var.value.model_dir }}/ {{ ds }}",
        network_mode="bridge",
        do_xcom_push=False,
        volumes=[DEFAULT_VOLUME]
    )
    end_task = DummyOperator(task_id='end-prediction')

    start_task >> preprocessing >> prediction >> end_task

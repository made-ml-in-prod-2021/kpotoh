from datetime import timedelta

DEFAULT_VOLUME = '/home/mr/MADE/DS-22/ml_in_prod/airflow_ml_dags/data:/data'

default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "email_on_failure": True,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

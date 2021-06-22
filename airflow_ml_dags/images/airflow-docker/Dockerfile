ARG AIRFLOW_BASE_IMAGE="apache/airflow:2.0.0-python3.6"
FROM ${AIRFLOW_BASE_IMAGE}

RUN pip install --user --no-cache-dir \
    apache-airflow-providers-docker==1.2.0 pytest==6.2.4

COPY test_dags.py /opt/airflow/tests/test_dags.py

USER root
ENV PYTHONPATH=/home/airflow/.local/lib/python3.6/site-packages

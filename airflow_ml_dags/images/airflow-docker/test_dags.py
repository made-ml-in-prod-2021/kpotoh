import os
import sys

import pytest
from airflow import DAG
from airflow.models import DagBag
import unittest

sys.path.append("dags")


class TestDAGs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dagbag = DagBag(dag_folder="dags/", include_examples=False)

    def assertDagDictEqual(self, source: dict, dag: DAG):
        assert dag.task_dict.keys() == source.keys()
        for task_id, downstream_list in source.items():
            assert dag.has_task(task_id)
            task = dag.get_task(task_id)
            assert task.downstream_task_ids == set(downstream_list)

    def test_dag_loaded(self):
        dags = self.dagbag.dags.values()
        assert len(dags) == 3
        for dag in dags:
            assert dag is not None
        assert self.dagbag.import_errors == {}

    def test_dag1_structure(self):
        dag1 = self.dagbag.get_dag(dag_id='DAG1_generate_data')
        structure1 = {
            "start-generation": ["generate-data"],
            "generate-data": ["end-generation"],
            "end-generation": [],
        }
        self.assertDagDictEqual(structure1, dag1)

    def test_dag2_structure(self):
        dag2 = self.dagbag.get_dag(dag_id='DAG2_train_model')
        structure2 = {
            "start-train-pipeline": ["await-target", "await-data"],
            "await-data": ["preprocessing"],
            "await-target": ["preprocessing"],
            "preprocessing": ["splitting"],
            "splitting": ["training"],
            "training": ["validation"],
            "validation": ["end-train-pipeline"],
            "end-train-pipeline": [],
        }
        self.assertDagDictEqual(structure2, dag2)

    def test_dag3_structure(self):
        dag3 = self.dagbag.get_dag(dag_id='DAG3_inference')
        structure3 = {
            "start-prediction": ["await-data", "await-model"],
            "await-data": ["preprocessing"],
            "await-model": ["preprocessing"],
            "preprocessing": ["prediction"],
            "prediction": ["end-prediction"],
            "end-prediction": [],
        }
        self.assertDagDictEqual(structure3, dag3)

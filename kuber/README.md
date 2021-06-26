Homework 4. Kubernetes games
----------------

### Commands

```
kubectl apply -f kubernetes_manifests/fastapi.yaml
kubectl get pods/rs
kubectl describe deployments
kubectl port-forward pod/fastapi-ml 8000:8000
```

## Tasks:

- [X] 0 Установите kubectl

- [X] 1) Разверните kubernetes - [minikube](https://minikube.sigs.k8s.io/docs/start/) **(5 баллов)**

- [X] 2) Напишите простой pod manifests для вашего приложения, назовите его online-inference-pod.yaml **(4 балла)**

- [X] 2а) Пропишите requests/limits и напишите зачем это нужно в описание PR; закоммитьте файл online-inference-pod-resources.yaml **(2 баллa)**

- [X] 3) Модифицируйте свое приложение так, чтобы оно стартовало не сразу(с задержкой секунд 20-30) и падало спустя минуты работы. Добавьте liveness и readiness пробы , посмотрите что будет происходить. Напишите в описании -- чего вы этим добились. Закоммититьте отдельный манифест online-inference-pod-probes.yaml (и изменение кода приложения). Опубликуйте ваше приложение(из ДЗ 2) с тэгом v2. **(3 балла)**

- [X] 4) Создайте replicaset, сделайте 3 реплики вашего приложения. Закоммитьте online-inference-replicaset.yaml. Ответьте на вопрос, что будет, если сменить докер образа в манифесте и одновременно с этим: а) уменьшить число реплик; б) увеличить число реплик.
Поды с какими версиями образа будут внутри будут в кластере? **(3 балла)**

- [X] 5) Опишите деплоймент для вашего приложения. Играя с параметрами деплоя(maxSurge, maxUnavaliable), добейтесь ситуации, когда при деплое новой версии: a) Есть момент времени, когда на кластере есть как все старые поды, так и все новые (опишите эту ситуацию) (закоммититьте файл online-inference-deployment-blue-green.yaml); б) одновременно с поднятием новых версии, гасятся старые (закоммитите файл online-inference-deployment-rolling-update.yaml); **(3 балла)**

## Run online inference from docker

~~~
docker build -t kpotoh/online_inference:v1 .
docker run -p 8000:8000 kpotoh/online_inference:v1
python make_request.py 
~~~
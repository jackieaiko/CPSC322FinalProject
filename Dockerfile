FROM continuumio/anaconda3:2020.11

ADD . /code
WORKDIR /code
ENTRYPOINT ["python", "random_forest_app.py"]
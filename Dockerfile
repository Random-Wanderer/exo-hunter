FROM python:3.8-buster

COPY api api
COPY ExoHunter ExoHunter
COPY rnn_model.joblib rnn_model.joblib
COPY requirements.txt requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn api.api:app --host 0.0.0.0 --port $PORT

FROM python:3.8-buster

COPY api api
COPY ExoHunter ExoHunter
COPY model_for_cnn_r88_p64.pkl model_for_cnn_r88_p64.pkl
COPY requirements.txt requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn api.api:app --host 0.0.0.0 --port $PORT

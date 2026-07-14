FROM python:3.10-slim

WORKDIR /code


COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt


RUN mkdir -p /tmp/flashrank_cache && chmod -R 777 /tmp/flashrank_cache


COPY . .


CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]

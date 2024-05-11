FROM python:3.8-slim-buster

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5555
RUN echo "${PWD}"
CMD ["flask", "--app", "src/webserver/app:app", "run", "--host=0.0.0.0", "--port=5555"]
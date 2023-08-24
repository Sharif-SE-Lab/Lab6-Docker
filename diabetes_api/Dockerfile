FROM python:3.8

RUN apt-get update -y

WORKDIR /app

COPY requirements.txt ./
RUN pip install -i https://pypi.iranrepo.ir/simple -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED 1

CMD ["/bin/bash","-c","./startup.sh"]

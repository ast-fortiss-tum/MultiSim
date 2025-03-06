FROM python:3.8-slim-buster

WORKDIR /src

COPY ./requirements.txt /src

RUN pip install --upgrade pip --trusted-host pypi.org --trusted-host files.pythonhosted.org
RUN pip install -r requirements.txt --trusted-host pypi.org --trusted-host files.pythonhosted.org

COPY . /src

ENTRYPOINT ["python", "-u", "run.py"]

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY . /code

CMD ["fastapi", "run", "--port", "8080"]
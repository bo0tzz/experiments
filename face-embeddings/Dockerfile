FROM python:3.10

WORKDIR /

ENV DEEPFACE_HOME="/cache"

RUN apt update && apt install -y libgl1-mesa-glx
RUN pip install deepface

COPY embeddings.py .

CMD ["python", "/embeddings.py"]
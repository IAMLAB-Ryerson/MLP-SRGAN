FROM pytorch/pytorch:latest

RUN apt-get -y update
RUN apt install libgl1-mesa-glx -y
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements.txt .
RUN pip3 install -r requirements.txt

WORKDIR /app
COPY app/ .

ENTRYPOINT ["python", "inference_mlp_srgan_codido.py"]

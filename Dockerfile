FROM python:3.6 as build

ENV LANG C.UTF-8

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        espeak libsndfile1 git

RUN mkdir -p /app
WORKDIR /app

# Clone repo
RUN git clone https://github.com/mozilla/TTS -b dev

# New virtualenv
RUN python3 -m venv .venv

# Install requirements
WORKDIR /app/TTS
RUN ../.venv/bin/pip3 install -U pip && \
    ../.venv/bin/pip3 install -r requirements.txt

# Extra packages missing from requirements
#RUN .venv/bin/pip3 install inflect 'numba==0.48'

# Packages needed for web server
RUN ../.venv/bin/pip3 install flask flask-cors

# Install TTS
RUN ../.venv/bin/python3 setup.py install

# -----------------------------------------------------------------------------

FROM python:3.6-slim

# Install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        espeak libsndfile1 && \
    apt-get clean

# Copy installed build
COPY --from=build /app/.venv/ /app/
# Copy models
COPY model/ /app/model/
COPY vocoder/ /app/vocoder/
# Copy code
COPY templates/ /app/templates/
COPY tts.py /app/

WORKDIR /app

EXPOSE 5002/tcp

ENTRYPOINT ["/app/bin/python3", "/app/tts.py"]
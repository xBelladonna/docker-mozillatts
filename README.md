# Mozilla TTS

Docker image for [Mozilla TTS](https://github.com/mozilla/TTS).

Includes [@erogol's](https://github.com/erogol) pre-built multi-speaker Tacotron2 English model and full-band MelGAN vocoder.
See [below](#building-yourself) for links to specific checkpoints.

## Using

Use synesthesiam's prebuilt image containing LJSpeech Tacotron2 model with the below command. Note that this model only has a single speaker, but is the highest quality pre-trained model available.

```sh
$ docker run -p 5002:5002 synesthesiam/mozillatts
```

Visit http://localhost:5002 for web interface.

Do HTTP GET at http://localhost:5002/api/tts?text=your%20sentence to get WAV audio back:

```sh
$ curl -G --output - \
    -H 'Content-Type: text/plain' \
    --data-urlencode 'text=Welcome to the world of speech synthesis!' \
    'http://localhost:5002/api/tts' | \
    aplay
```

Optionally if building and using the multi-speaker model (see [Building Yourself](#building-yourself)), `speaker` can be specified as another URL encoded argument. This argument takes an integer (or string) which is the index of the speaker whose embeddings are loaded into the model, i.e. the voice the model uses to synthesize speech.

```sh
$ curl -G --output - \
    -H 'Content-Type: text/plain' \
    --data-urlencode 'text=Hello World!' \
    --data-urlencode 'speaker=7' \
    'http://localhost:5002/api/tts' | \
    aplay
```

The API also takes POST requests with JSON data with the following structure:

```json
{
    "text": "Your text here",
    "speaker": 7
}
```

Example:

```sh
$ curl --output - \
    -H 'Content-Type: application/json' \
    --data '{"text": "Hello World!", "speaker": 7}' \
    'http://localhost:5002/api/tts' | \
    aplay
```

Again, the speaker value is optional and doesn't need to be given, it will use the first speaker in the dataset

## Building Yourself

The Docker image is built using a custom implementation based on [this code](https://colab.research.google.com/drive/1t0TFC3vqU1nFow5p5FTPjtkT6rFJOSsB?usp=sharing#scrollTo=r0IEFZ0B5vQg). You'll need to manually download the model and vocoder checkpoints/configs:

* [`model/config.json`](https://drive.google.com/uc?id=1YKrAQKBLVXzyYS0CQcLRW_5eGfMOIQ-2)
* [`model/speakers.json`](https://drive.google.com/uc?id=1oOnPWI_ho3-UJs3LbGkec2EZ0TtEOc_6)
* [`model/best_model.pth.tar`](https://drive.google.com/uc?id=1iDCL_cRIipoig7Wvlx4dHaOrmpTQxuhT)
* [`vocoder/config.json`](https://drive.google.com/uc?id=1BmaZ2tOJZLrGGnjOEEjuIw9KPHA43vgC)
* [`vocoder/checkpoint_1450000.pth.tar`](https://drive.google.com/uc?id=1DX9ZMfCxmzGnL9dmnf98V9mVVUNRnY1j)

1. Place the model files in a folder called `model` and the vocoder files in a folder called `vocoder`, both in this repo directory.
2. Build the image: `docker build . -t mozillatts`
3. Run a container: `docker run -p 5002:5002 -e PYTHONUNBUFFERED=TRUE mozillatts`
4. Navigate to http://localhost:5002
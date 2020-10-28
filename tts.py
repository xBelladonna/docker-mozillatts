#!/usr/bin/env python3

import os
import argparse
import io
import json
import time
import torch
import numpy as np

from flask import Flask, Response, render_template, request
from flask_cors import CORS

from TTS.utils.io import load_config
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.generic_utils import setup_model
from TTS.tts.utils.text.symbols import symbols, phonemes, make_symbols
from TTS.tts.utils.synthesis import synthesis
from TTS.tts.utils.io import load_checkpoint
from TTS.vocoder.utils.generic_utils import setup_generator

# -----------------------------------------------------------------------------


def tts(model, vocoder_model, text, CONFIG, USE_CUDA: bool, ap, USE_GL: bool, SPEAKER_FILEID=None, speaker_embedding=None, STYLE_WAV=None):
    t_1 = time.time()
    waveform, _, _, mel_postnet_spec, _, _ = synthesis(
        model, text, CONFIG, USE_CUDA, ap, SPEAKER_FILEID, STYLE_WAV, False, CONFIG.enable_eos_bos_chars, USE_GL, speaker_embedding=speaker_embedding)

    mel_postnet_spec = ap._denormalize(mel_postnet_spec.T).T

    if not USE_GL:
        vocoder_input = ap_vocoder._normalize(mel_postnet_spec.T)
        if scale_factor[1] != 1:
            vocoder_input = interpolate_vocoder_input(
                scale_factor, vocoder_input)
        else:
            vocoder_input = torch.tensor(vocoder_input).unsqueeze(0)
        waveform = vocoder_model.inference(vocoder_input)
    if USE_CUDA and not USE_GL:
        waveform = waveform.cpu()
    if not USE_GL:
        waveform = waveform.numpy()
    waveform = waveform.squeeze()

    rt = time.time() - t_1
    rtf = (time.time() - t_1) / (len(waveform) / ap.sample_rate)
    tps = (time.time() - t_1) / len(waveform)
    print(f" > Run time: {round(rt, 2)}s")
    print(f" > Real-time factor: {rtf}")
    print(f" > Time per step: {tps}s")

    return waveform


def interpolate_vocoder_input(scale_factor, spec):
    """Interpolation to tolarate the sampling rate difference
    btw tts model and vocoder"""
    print(" > Before interpolation: ", spec.shape)
    spec = torch.tensor(spec).unsqueeze(0).unsqueeze(0)
    spec = torch.nn.functional.interpolate(
        spec, scale_factor=scale_factor, mode='bilinear').squeeze(0)
    print(" > After interpolation: ", spec.shape)
    return spec


def extract_speaker_names(speaker_json):
    speaker_names = json.load(open(speaker_json, 'r'))
    speakers = []
    last_name = ""

    for key in speaker_names.keys():
        name = speaker_names[key]["name"]
        if name != last_name:
            speakers.append(speaker_names[key]["name"])
        last_name = name

    return speakers

# -----------------------------------------------------------------------------

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--speaker", type=int, default=None,
                    help="Index number of speaker name. TTS will use this speaker's embeddings to generate speech. Inspect speakers.json for names.")
args = parser.parse_args()

# Runtime settings
USE_CUDA = False

# Model paths
_DIR = os.path.dirname(os.path.realpath(__file__))

TTS_MODEL = os.path.join(_DIR, "model", "best_model.pth.tar")
TTS_CONFIG = os.path.join(_DIR, "model", "config.json")
VOCODER_MODEL = os.path.join(_DIR, "vocoder", "checkpoint_1000000.pth.tar")
VOCODER_CONFIG = os.path.join(_DIR, "vocoder", "config.json")
SPEAKER_JSON = os.path.join(_DIR, "model", "speakers.json")
SPEAKER_FILEID = None  # "p269_215.wav"  # if None use the first embedding from speakers.json
STYLE_WAV = None # filename of encoder preprocessed wav or GST tokens as dict

# Load configs
TTS_CONFIG = load_config(TTS_CONFIG)
VOCODER_CONFIG = load_config(VOCODER_CONFIG)

# Load the audio processor
ap = AudioProcessor(**TTS_CONFIG.audio)

# Set config options
TTS_CONFIG.forward_attn_mask = True

# if the vocabulary was passed, replace the default
if 'characters' in TTS_CONFIG.keys():
    symbols, phonemes = make_symbols(**TTS_CONFIG.characters)

# Get list of speaker names from speakers.json
train_speakers = extract_speaker_names(SPEAKER_JSON)
num_speakers = len(train_speakers)
print(f" > Speakers available: {num_speakers}")

# Choose speaker
speaker_index = 0
SPEAKER_CHOICE = train_speakers[speaker_index]

if args.speaker is not None:
    speaker_index = args.speaker
    SPEAKER_CHOICE = train_speakers[speaker_index]
    print(f"Speaker #{args.speaker} ({SPEAKER_CHOICE}) selected")
else:
    print(f" > Speaker #{speaker_index} ({SPEAKER_CHOICE}) has been chosen automatically")

num_samples_speaker = 100
speaker_embedding = None
speaker_embedding_dim = None
num_speaker_embeddings = 0

# Load speaker
if SPEAKER_JSON != '':
    speaker_mapping = json.load(open(SPEAKER_JSON, 'r'))
    num_speaker_embeddings = len(speaker_mapping)
    if TTS_CONFIG.use_external_speaker_embedding_file:
        if SPEAKER_FILEID is not None:
            speaker_embedding = speaker_mapping[SPEAKER_FILEID]['embedding']
        elif SPEAKER_CHOICE is not None:
            if TTS_CONFIG.use_external_speaker_embedding_file:
                speaker_embeddings = []
                for key in list(speaker_mapping.keys()):
                    if SPEAKER_CHOICE in speaker_mapping[key]['name']:
                        if len(speaker_embeddings) < num_samples_speaker:
                            speaker_embeddings.append(
                                speaker_mapping[key]['embedding'])
                # takes the average of the embeddings samples of the speakers
                speaker_embedding = np.mean(
                    np.array(speaker_embeddings), axis=0).tolist()
        else:  # if SPEAKER_FILEID or SPEAKER_CHOICE is not specified use the first sample in speakers.json
            choice_speaker = list(speaker_mapping.keys())[0]
            print(" Speaker ", choice_speaker.split(
                '_')[0], "was chosen automatically")
            speaker_embedding = speaker_mapping[choice_speaker]['embedding']
        speaker_embedding_dim = len(speaker_embedding)

# Load TTS model
num_chars = len(phonemes) if TTS_CONFIG.use_phonemes else len(symbols)
model = setup_model(num_chars, num_speaker_embeddings, TTS_CONFIG, speaker_embedding_dim)
model, _ = load_checkpoint(model, TTS_MODEL, use_cuda=USE_CUDA)
if USE_CUDA:
    model.cuda()
model.eval()

# Load vocoder model
USE_GL = False # Don't use Griffin Lim if vocoder is used
vocoder_model = setup_generator(VOCODER_CONFIG)
vocoder_model.load_state_dict(torch.load(
    VOCODER_MODEL, map_location="cpu")["model"])
vocoder_model.remove_weight_norm()
vocoder_model.inference_padding = 0

# Scale factor for sampling rate difference
scale_factor = [1,  VOCODER_CONFIG['audio']['sample_rate'] / ap.sample_rate]
print(f" > Model sample rate: {ap.sample_rate}")
print(f" > Vocoder sample rate: {VOCODER_CONFIG['audio']['sample_rate']}")
print(f" > Scale factor: {scale_factor}")

ap_vocoder = AudioProcessor(**VOCODER_CONFIG['audio'])
if USE_CUDA:
    vocoder_model.cuda()
vocoder_model.eval()

# -----------------------------------------------------------------------------

app = Flask("mozillatts")
CORS(app)

@app.route("/api/tts", methods=['GET', 'POST'])
def api_tts():
    global num_speakers
    global speaker_embedding
    global SPEAKER_CHOICE

    if request.method == 'GET':
        text = request.args.get("text")
        new_speaker_index = request.args.get("speaker")
    else:
        content = request.get_json()

        text = content["text"]
        new_speaker_index = content["speaker"]

    if text is None:
        return Response("No text provided", status=400)

    if new_speaker_index is not None and new_speaker_index is not "":
        # arrays are zero-indexed so we return 400 if equal to or greater than the number of elements in the array
        if int(new_speaker_index) >= num_speakers:
            return Response(f"Speaker does not exist. There are only {num_speakers} speakers in total. Please select a speaker from 0-{num_speakers - 1}", status=400)

        NEW_SPEAKER_CHOICE = train_speakers[int(new_speaker_index.strip())]
        print(
            f" > Speaker #{new_speaker_index} ({NEW_SPEAKER_CHOICE}) selected")

        new_speaker_embeddings = []
        for key in list(speaker_mapping.keys()):
            if NEW_SPEAKER_CHOICE in speaker_mapping[key]['name']:
                if len(new_speaker_embeddings) < num_samples_speaker:
                    new_speaker_embeddings.append(
                        speaker_mapping[key]['embedding'])
        # takes the average of the embeddings samples of the speakers
        new_speaker_embedding = np.mean(
            np.array(new_speaker_embeddings), axis=0).tolist()

        wav = tts(model, vocoder_model, text.strip(), TTS_CONFIG, USE_CUDA, ap, USE_GL,
                    SPEAKER_FILEID, speaker_embedding=new_speaker_embedding, STYLE_WAV=STYLE_WAV)
    else:
        print(f" > Using default speaker #{speaker_index} ({SPEAKER_CHOICE})")
        wav = tts(model, vocoder_model, text.strip(), TTS_CONFIG, USE_CUDA, ap, USE_GL,
                    SPEAKER_FILEID, speaker_embedding=speaker_embedding, STYLE_WAV=STYLE_WAV)

    with io.BytesIO() as out:
        ap.save_wav(wav, out)
        return Response(out.getvalue(), mimetype="audio/wav")


@app.route("/")
def index():
    return render_template("index.html")


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)

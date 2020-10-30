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


# Main generator function
def tts(model, vocoder_model, text, CONFIG, use_cuda: bool, ap, use_gl: bool, SPEAKER_FILEID=None, speaker_embedding=None, gst_style=None):
    t_1 = time.time()
    waveform, _, _, mel_postnet_spec, _, _ = synthesis(
        model, text, CONFIG, use_cuda, ap, SPEAKER_FILEID, gst_style, False, CONFIG.enable_eos_bos_chars, use_gl, speaker_embedding=speaker_embedding)

    mel_postnet_spec = ap._denormalize(mel_postnet_spec.T).T

    if not use_gl:
        vocoder_input = ap_vocoder._normalize(mel_postnet_spec.T)
        if scale_factor[1] != 1:
            vocoder_input = interpolate_vocoder_input(
                scale_factor, vocoder_input)
        else:
            vocoder_input = torch.tensor(vocoder_input).unsqueeze(0)
        waveform = vocoder_model.inference(vocoder_input)
    if use_cuda and not use_gl:
        waveform = waveform.cpu()
    if not use_gl:
        waveform = waveform.numpy()
    waveform = waveform.squeeze()

    rt = time.time() - t_1
    rtf = (time.time() - t_1) / (len(waveform) / ap.sample_rate)
    tps = (time.time() - t_1) / len(waveform)
    print(f" > Run time: {round(rt, 2)}s")
    print(f" > Real-time factor: {rtf}")
    print(f" > Time per step: {tps}s")

    return waveform


# Utility functions
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
    speaker_names = []
    last_name = ""

    for key in speaker_json.keys():
        name = speaker_json[key]["name"]
        if name != last_name:
            speaker_names.append(speaker_json[key]["name"])
        last_name = name

    return speaker_names


def build_speaker_embedding(speaker_choice, speaker_mapping, num_samples):
    speaker_embeddings = []
    num_embeddings = 0
    print(" > Building speaker embedding...")
    for key in list(speaker_mapping.keys()):
        if speaker_choice in speaker_mapping[key]['name']:
            if len(speaker_embeddings) < num_samples:
                speaker_embeddings.append(speaker_mapping[key]['embedding'])
                num_embeddings += 1
    # takes the average of the embeddings samples of the speaker
    speaker_embedding = np.mean(
        np.array(speaker_embeddings), axis=0).tolist()
    if num_samples > num_embeddings:
        print(f" > WARNING: Number of embeddings was truncated due to desired number of samples ({num_samples}) being greater than the number of embeddings available")
    print(f" > Averaged {num_embeddings} embeddings from speaker {speaker_choice}")

    return speaker_embedding

# -----------------------------------------------------------------------------


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--speaker", type=int, default=None,
                    help="Index number of speaker name. TTS will use the average of a subset of this speaker's embeddings (specified with --speaker-num-samples) to generate speech. Inspect speakers.json for names.")
parser.add_argument("--speaker-num-samples", type=int, default=2,
                    help="Number of embeddings to average if selecting a speaker with --speakers. The higher the value, the closer to the original voice. Defaults to 2.")
parser.add_argument("--speaker-file-id", type=str, default=None,
                    help="Name of specific speaker embedding file. Overrides --speaker. If specified, this specific embedding will be used instead of averaging the embeddings of a particular speaker.")
parser.add_argument("--gst-style", type=str, default=None,
                    help="Filename of your GST reference wav file or JSON-formatted token dictionary")

parser.add_argument("--use-cuda", action="store_true",
                    help="Uses CUDA backend if flag is passed")

parser.add_argument("--tts-model", type=str, default="best_model.pth.tar",
                    help="Name of the JSON-formatted TTS model config file. Defaults to config.json")
parser.add_argument("--tts-config", type=str, default="config.json",
                    help="Name of the pre-trained TTS model file. Defaults to best_model.pth.tar")
parser.add_argument("--speaker-embeddings", type=str, default="speakers.json",
                    help="Name of the JSON-formatted speaker embeddings file. Defaults to speakers.json")
parser.add_argument("--vocoder-model", type=str, default="checkpoint_1000000.pth.tar",
                    help="Name of the pre-trained vocoder model file. Defaults to checkpoint_1000000.pth.tar")
parser.add_argument("--vocoder-config", type=str, default="config.json",
                    help="Name of the JSON-formatted vocoder model config file. Defaults to config.json")
args = parser.parse_args()

# -----------------------------------------------------------------------------

# Runtime settings
use_cuda = True if args.use_cuda else False

# Model paths
_DIR = os.path.dirname(os.path.realpath(__file__))

TTS_MODEL = os.path.join(_DIR, "model", args.tts_model)
TTS_CONFIG = os.path.join(_DIR, "model", args.tts_config)
VOCODER_MODEL = os.path.join(_DIR, "vocoder", args.vocoder_model)
VOCODER_CONFIG = os.path.join(_DIR, "vocoder", args.vocoder_config)
SPEAKER_JSON = os.path.join(_DIR, "model", args.speaker_embeddings)
SPEAKER_FILEID = args.speaker_file_id  # e.g. p269_215.wav  # if None use embeddings from speaker in speaker_choice if given or first speaker available

# Load configs
TTS_CONFIG = load_config(TTS_CONFIG)
VOCODER_CONFIG = load_config(VOCODER_CONFIG)

# Load the audio processor
ap = AudioProcessor(**TTS_CONFIG.audio)

# Set config options
#TTS.CONFIG.use_forward_attn = True  # If it uses forward attention it aligns faster in general
#TTS_CONFIG.forward_attn_mask = True  # Additional masking forcing monotonicity (and a correlative slight increase in rate)

# if the vocabulary was passed, replace the default
if 'characters' in TTS_CONFIG.keys():
    symbols, phonemes = make_symbols(**TTS_CONFIG.characters)

# Load GST style
if args.gst_style is not None:
    gst_style_path = os.path.join(_DIR, "model", args.gst_style)

    if args.gst_style.endswith(".wav"):
        gst_style = gst_style_path
        print(f" > GST style: Extracting from {gst_style}")
    else:
        gst_style = json.load(open(gst_style_path))
        print(f" > GST style: {json.dumps(gst_style)}")
else:
    gst_style = None

# Set up speaker embedding
speaker_embedding = None
speaker_embedding_dim = None
num_speaker_embeddings = 0

# Load speaker embeddings
print(" > Loading speaker mapping...")
speaker_mapping = json.load(open(SPEAKER_JSON, 'r'))
num_speaker_embeddings = len(speaker_mapping)

# Get list of speaker names from speaker embeddings
speaker_names = extract_speaker_names(speaker_mapping)
num_speakers = len(speaker_names)
print(f" > Speakers available: {num_speakers}")

if os.path.exists(SPEAKER_JSON):
    if TTS_CONFIG.use_external_speaker_embedding_file:
        # Choose speaker
        if SPEAKER_FILEID is not None:
            speaker_embedding = speaker_mapping[SPEAKER_FILEID]['embedding']
            print(f" > Using embedding {speaker_embedding.split('_')[1].split('.wav')[0]} of speaker {speaker_embedding.split('_')[0]}")
        else:
            if args.speaker is not None:
                speaker_index = args.speaker
                speaker_choice = speaker_names[speaker_index]
                print(f" > Speaker #{args.speaker} ({speaker_choice}) selected")
            else:
                speaker_index = 0
                speaker_choice = speaker_names[speaker_index]
                print(f" > Speaker #{speaker_index} ({speaker_choice}) has been chosen automatically")

            speaker_embedding = build_speaker_embedding(speaker_choice, speaker_mapping, args.speaker_num_samples)

        speaker_embedding_dim = len(speaker_embedding)
    else:
        print(" > TTS config specifies not to use external speaker embedding file. No speaker embedding loaded.")

# Load TTS model
num_chars = len(phonemes) if TTS_CONFIG.use_phonemes else len(symbols)
model = setup_model(num_chars, num_speaker_embeddings,
                    TTS_CONFIG, speaker_embedding_dim)
model, _ = load_checkpoint(model, TTS_MODEL, use_cuda=use_cuda)
if use_cuda:
    model.cuda()
model.eval()

# Load vocoder model
use_gl = False  # Don't use Griffin Lim if vocoder is used
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
if use_cuda:
    vocoder_model.cuda()
vocoder_model.eval()

# -----------------------------------------------------------------------------

app = Flask("mozillatts")
CORS(app)


@app.route("/api/tts", methods=['GET', 'POST'])
def api_tts():
    if request.method == 'GET':
        text = request.args.get("text")
        _speaker_index = request.args.get("speaker")
    else:
        content = request.get_json()
        keys = content.keys()

        text = content["text"]
        _speaker_index = content["speaker"] if "speaker" in keys else None

    if text is None:
        return Response("No text provided", status=400)

    if _speaker_index is not None and _speaker_index is not "":
        # arrays are zero-indexed so we return 400 if equal to or greater than the number of elements in the array
        if int(_speaker_index) >= num_speakers:
            return Response(f"Speaker does not exist. There are only {num_speakers} speakers in total. Please select a speaker from 0-{num_speakers - 1}", status=400)

        _speaker_choice = speaker_names[int(_speaker_index.strip())]
        print(f" > Speaker #{_speaker_index} ({_speaker_choice}) selected")
        _speaker_embedding = build_speaker_embedding(_speaker_choice, speaker_mapping, args.speaker_num_samples)
    else:
        _speaker_embedding = speaker_embedding

    wav = tts(model, vocoder_model, text.strip(), TTS_CONFIG, use_cuda, ap, use_gl,
              SPEAKER_FILEID, speaker_embedding=_speaker_embedding, gst_style=gst_style)

    with io.BytesIO() as out:
        ap.save_wav(wav, out)
        return Response(out.getvalue(), mimetype="audio/wav")


@app.route("/")
def index():
    return render_template("index.html")


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)

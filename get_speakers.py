import os
import json

_DIR = os.path.dirname(os.path.realpath(__file__))

speakers = []

speakers_file = open(os.path.join(_DIR, "model", "speakers.json"), 'r')
speaker_names = json.load(speakers_file)
last_key = ""

for key in speaker_names.keys():
    name = speaker_names[key]["name"]
    if name != last_key:
        speakers.append(speaker_names[key]["name"])
    last_key = name

data = {"speaker_total": len(speakers), "speakers": speakers}

print(json.dumps(data))

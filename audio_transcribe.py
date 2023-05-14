import numpy as np
from sklearn.cluster import AgglomerativeClustering
import contextlib
import wave
from time import time
from pyannote.core import Segment
from pyannote.audio import Audio
import whisper
import datetime
import subprocess
from tqdm import tqdm
import torch
from utils.utils import time
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

embedding_model = PretrainedSpeakerEmbedding(
    "speechbrain/spkrec-ecapa-voxceleb",
    device=torch.device("cpu")
    )

'''
Set paramters
'''
num_speakers = None  # @param {type:"integer"}
language = 'English'  # @param ['any', 'English']
model_size = 'large'  # @param ['tiny', 'base', 'small', 'medium', 'large']
model_name = model_size


if language == 'English' and model_size != 'large':
    model_name += '.en'

'''
Set audio paths
'''
audio_name = "Born_to_MISrepresent_EP7p1.mp3"
audio_path = "audio/" + audio_name


'''
Convert to wav
'''
# convert to wav
if audio_path.split('.')[-1] != 'wav':
    new_audio_path = "audio/" + audio_name.split('.')[0] + '.wav'
    subprocess.call(['ffmpeg', '-i', audio_path, new_audio_path, '-y'],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    audio_path = new_audio_path

print("Loading model...")
model = whisper.load_model(model_size)

print("Transcribing audio...")
s = time()
result = model.transcribe(audio_path)
e = time()
print(f"Finished transcribing. {np.round((e-s), 2)}s ")


segments = result["segments"]

with contextlib.closing(wave.open(audio_path, 'r')) as f:
    frames = f.getnframes()
    rate = f.getframerate()
    duration = frames / float(rate)

audio = Audio()


def segment_embedding(segment):
    start = segment["start"]
    # Whisper overshoots the end timestamp in the last segment
    end = min(duration, segment["end"])
    clip = Segment(start, end)
    waveform, _ = audio.crop(audio_path, clip)
    return embedding_model(waveform[None])


print("Segmenting audio...")
embeddings = np.zeros(shape=(len(segments), 192))
for i, segment in tqdm(enumerate(segments)):
    embeddings[i] = segment_embedding(segment)

embeddings = np.nan_to_num(embeddings)

'''
Cluster audio embeddings. 
TODO: Test with dbscan for no-parameter clustering
'''
clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
labels = clustering.labels_
for i in range(len(segments)):
    segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)

f = open(f"transcriptions/{audio_name.split('.')[0]}_transcript.txt", "w")
for (i, segment) in enumerate(segments):
    if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
        f.write("\n" + segment["speaker"] + ' ' +
                str(time(segment["start"])) + '\n')
    f.write(segment["text"][1:] + ' ')
f.close()

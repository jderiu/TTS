import random

from pydub import AudioSegment
import torch, os
from TTS.api import TTS

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "GPT_XTTS_v2.0_LJSpeech_FT-November-27-2024_09+16PM-afa6889d"

model_path = f"/cluster/data/deri/TTS/TTS_dante/trained/{model_name}/"
config_path = f"/cluster/data/deri/TTS/TTS_dante/trained/{model_name}/config.json"
# Init TTS
tts = TTS(
    model_path=model_path,
    config_path=config_path,
    progress_bar=True
).to(device)

tts.synthesizer.seg = tts.synthesizer._get_segmenter('it')
# Run TTS
# ❗ Since this model is multi-lingual voice cloning model, we must set the target speaker_wav and language
# Text to speech list of amplitude values as output
#wav = tts.tts(text="Hello world!", speaker_wav="my/cloning/audio.wav", language="en")
# Text to speech to a file

speaker_wavs_base = 'D:/02Datasets/02 Audio Processing/dante/inferno/audio_segment_clean/canto_1'
speaker_wavs = [os.path.join(speaker_wavs_base, x) for x in os.listdir(speaker_wavs_base) if 'terzina' in x]

path = 'D:/02Datasets/02 Audio Processing/dante/purgatorio'

with open(os.path.join(path, 'terzine_manual/canto_1.txt'), 'rt', encoding='utf-8') as file:
    lines = file.read()
    for idx, line in enumerate(lines.split('\n')):
        tts.tts_to_file(text=line, speaker_wav=[random.choice(speaker_wavs) for _ in range(3)], language="it", split_sentences=False, file_path=os.path.join(path, f'pred_audio/canto_1/audio_{idx}.wav'))



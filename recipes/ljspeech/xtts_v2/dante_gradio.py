import gradio as gr
import torch

from TTS.api import TTS

import os
import librosa
import soundfile as sf

# import whisperx
# from huggingface_hub import hf_hub_download

# print("Loading ECAPA2 model...")
# # automatically checks for cached file, optionally set `cache_dir` location
# model_file = hf_hub_download(repo_id='Jenthe/ECAPA2', filename='ecapa2.pt', cache_dir=None)
#
# ecapa2 = torch.jit.load(model_file, map_location='cuda')
# ecapa2.half() # optional, but results in faster inference
#
# print("Loading model...")
# device = "cuda"
# compute_type = "int8"
# whipser_model = whisperx.load_model("large-v3", device="cuda", compute_type=compute_type)


SPEAKER_WAV_BPATH = 'tmp/speaker_wavs'
SPEAKER_RAW_WAV_BPATH = 'tmp/speaker_raw_wavs'
OUTPUT_WAV_BPATH = 'tmp/generated_audios'

os.makedirs(SPEAKER_WAV_BPATH, exist_ok=True)
os.makedirs(SPEAKER_RAW_WAV_BPATH, exist_ok=True)
os.makedirs(OUTPUT_WAV_BPATH, exist_ok=True)

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "GPT_XTTS_v2.0_LJSpeech_FT-April-26-2025_04+27PM-afa6889d"

model_path = f"/cluster/data/deri/TTS/TTS_dante/trained/{model_name}/"
config_path = f"/cluster/data/deri/TTS/TTS_dante/trained/{model_name}/config.json"

tts = TTS(
    model_path=model_path,
    config_path=config_path,
    progress_bar=True,
).to(device)
tts.eval()

tts.synthesizer.output_sample_rate = 24000


def generate_speech(
        text,
        length_penalty,
        beam_size,
        repetition_penalty,
        speaker_wav1,
        speaker_wav2,
        speaker_wav3,
):
    # save speaker wav into tmp/speaker_wavs with increasing numbering
    # get max number in speaker_wavs if any and increment by 1
    speaker_wav_path1 = os.path.join(SPEAKER_WAV_BPATH, f'tmp1.wav')
    speaker_wav_path2 = os.path.join(SPEAKER_WAV_BPATH, f'tmp2.wav')
    speaker_wav_path3 = os.path.join(SPEAKER_WAV_BPATH, f'tmp3.wav')
    out_wav_path = os.path.join(OUTPUT_WAV_BPATH, f'tmp.wav')

    sample_rate1, mono_data1 = speaker_wav1
    sample_rate2, mono_data2 = speaker_wav2
    sample_rate3, mono_data3 = speaker_wav3
    sf.write(speaker_wav_path1, mono_data1, sample_rate1)
    sf.write(speaker_wav_path2, mono_data2, sample_rate2)
    sf.write(speaker_wav_path3, mono_data3, sample_rate3)

    speaker_wav_paths = [
        speaker_wav_path1,
        speaker_wav_path2,
        speaker_wav_path3
    ]

    tts.tts_to_file(
        text=text,
        speaker_wav=speaker_wav_paths,
        language='it',
        split_sentences=False,
        file_path=out_wav_path,
        length_penalty=float(length_penalty),
        num_beams=int(beam_size),
        repetition_penalty=float(repetition_penalty),
    )
    # #load audio file
    # back_trans_audio, _ = torchaudio.load(out_wav_path)
    # speaker_tran_audio = librosa.load(speaker_wav_path, sr=16_000)
    #
    # result = whipser_model.transcribe(audio, batch_size=4, chunk_size=15, language=lang_tag, print_progress=True)
    # #concet all segments
    # transcript = []
    # for segment in result['segments']:
    #     transcript.append(segment['text'])
    # transcript = ' '.join(transcript)
    #
    # speaker_audio, _ = torchaudio.load(speaker_wav_path)  # sample rate of 16 kHz expected
    # embedding1 = ecapa2(speaker_audio.to('cuda'))
    # embedding2 = ecapa2(back_trans_audio.to('cuda'))
    #
    # cosine_sim = torch.nn.functional.cosine_similarity(embedding1, embedding2)

    ret_audio, sr = librosa.load(out_wav_path, sr=None)

    return sr, ret_audio  # ), transcript, cosine_sim


app = gr.Interface(
    fn=generate_speech,
    inputs=[
        gr.Textbox(lines=4, label="text"),
        gr.Number(value=1.0, label="length_penalty"),
        gr.Number(value=1, label="beam_size"),
        gr.Number(value=2.0, label="repetition_penalty"),
        gr.Audio(label="speaker_wav1"),
        gr.Audio(label="speaker_wav2"),
        gr.Audio(label="speaker_wav3"),
    ],
    outputs="audio")
app.launch()

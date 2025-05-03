import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import os, json, argparse
import pandas as pd
from tqdm import tqdm
from ch_test import prepare_unique_sentences, LANG_MAP, LANG_MAP_INV

def main(config):
    device = "cuda"

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

    model_name = config["model_name"]
    output_bpath = os.path.join(config["output_bpath"], model_name)
    dataspeech_stats_path = config["dataspeech_stats_path"]
    dataspeech_stats_fname = config["dataspeech_stats_fname"]
    test_sentence_path = config["test_sentence_path"]
    test_sentence_fname = config["test_sentnece_fname"]
    n_samples = config["n_samples"]

    dataspeech_path = os.path.join(dataspeech_stats_path, dataspeech_stats_fname)
    test_sentence_full_path = os.path.join(test_sentence_path, test_sentence_fname)

    os.makedirs(output_bpath, exist_ok=True)

    generate_kwargs = {
        "max_new_tokens": 64,
        "num_beams": 1,
        "no_repeat_ngram_size": 3,
        "condition_on_prev_tokens": True,
        "compression_ratio_threshold": 1.35,  # zlib compression ratio threshold (in token space)
        "temperature": (0.0, 0.2, 0.4),
        "logprob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "language": "german",
    }

    #laod randffile
    randdf = pd.read_csv(dataspeech_path, sep='\t')
    unique_test_sentence_df = prepare_unique_sentences(test_sentence_full_path, k=n_samples)

    unique_speakers = randdf['speaker_id'].unique()
    output_data = []
    for sid, speaker in enumerate(unique_speakers):
        opath_speaker = os.path.join(output_bpath, speaker)
        orig_dialect = randdf[randdf['speaker_id'] == speaker]['dialect'].iloc[0]
        orig_dialect_tag = LANG_MAP_INV[orig_dialect]
        for idx, row in tqdm(
                unique_test_sentence_df.iterrows(),
                total=len(unique_test_sentence_df),
                desc=f"Speaker {sid}/{len(unique_speakers)}: {speaker}"
        ):
            line = row['sentence']
            audio_files = []
            for dial_tag in LANG_MAP.keys():
                audio_file = os.path.join(opath_speaker, dial_tag, f'sent-{idx}.wav')
                if not os.path.exists(audio_file):
                    print(f"Skipping {opath_speaker}-{dial_tag}-{idx}")
                    continue
                audio_files.append((dial_tag, audio_file))
            if len(audio_files) == 0:
                print(f"Skipping {opath_speaker}-{idx}")
                continue
            results = pipe([x[1] for x in audio_files], generate_kwargs=generate_kwargs, batch_size=len(audio_files))
            for result, (dial_tag, audio_file) in zip(results, audio_files):
                back_translation = result["text"]
                output_data.append({
                    "speaker": speaker,
                    "sent_id": idx,
                    "sentence": line,
                    "back_translation": back_translation,
                    "audio_file": audio_file,
                    "dial_tag": dial_tag,
                    "orig_dialect_tag": orig_dialect_tag,
                })

    output_df = pd.DataFrame(output_data)
    output_df.to_csv(os.path.join(output_bpath, 'back_translation.tsv'), sep='\t', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='ProgramName',
        description='What the program does',
        epilog='Text at the bottom of help')

    parser.add_argument('-c', '--config', type=str, default='gen_config')
    args = parser.parse_args()

    with open(f'{args.config}.json', 'rt', encoding='utf-8') as f:
        config = json.load(f)

    main(config)
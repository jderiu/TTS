import argparse

import torch, os, json
from TTS.api import TTS

from tqdm import tqdm
import pandas as pd


LANG_MAP = {
    'ch_be': 'Bern',
    'ch_bs': 'Basel',
    'ch_gr': 'Graubünden',
    'ch_in': 'Innerschweiz',
    'ch_os': 'Ostschweiz',
    'ch_vs': 'Wallis',
    'ch_zh': 'Zürich',
    'de': 'Deutsch',
}
LANG_MAP_INV = {v:k for k,v in LANG_MAP.items()}

def prepare_unique_sentences(test_sentence_fname, k=50):
    unique_test_sentence_df = pd.read_csv(test_sentence_fname, sep='\t')
    # remove all sample_ids that are in the randdf
    if k > 0:
        unique_test_sentence_df = unique_test_sentence_df.head(k)

    return unique_test_sentence_df


def main(config):
    model_bpath = config["model_bpath"]
    model_name = config["model_name"]
    output_bpath = os.path.join(config["output_bpath"], model_name)
    dataspeech_stats_path = config["dataspeech_stats_path"]
    dataspeech_stats_fname = config["dataspeech_stats_fname"]
    speaker_ref_path = config["speaker_ref_path"]
    test_sentence_path = config["test_sentence_path"]
    test_sentence_fname = config["test_sentnece_fname"]
    n_samples = config["n_samples"]

    # Constructing full paths
    model_path = os.path.join(model_bpath, model_name)
    config_path = os.path.join(model_bpath, model_name, 'config.json')
    dataspeech_path = os.path.join(dataspeech_stats_path, dataspeech_stats_fname)
    test_sentence_full_path = os.path.join(test_sentence_path, test_sentence_fname)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(output_bpath, exist_ok=True)

    #laod randffile
    randdf = pd.read_csv(dataspeech_path, sep='\t')
    unique_test_sentence_df = prepare_unique_sentences(test_sentence_full_path, k=n_samples)

    unique_test_sentence_df.to_csv(os.path.join(output_bpath, 'unique_test_sentences.tsv'), sep='\t', index=False)
    randdf.to_csv(os.path.join(output_bpath, 'randdf.tsv'), sep='\t', index=False)

    # Init TTS
    tts = TTS(
        model_path=model_path,
        config_path=config_path,
        progress_bar=True
    ).to(device)

    unique_speakers = randdf['speaker_id'].unique()
    for sid, speaker in enumerate(unique_speakers):
        speaker_df = randdf[randdf['speaker_id'] == speaker]
        sample_ids = speaker_df['sample_id'].tolist()
        conditioning_paths = [os.path.join(speaker_ref_path, speaker, f"{audio}.wav") for audio in sample_ids]
        opath_speaker = os.path.join(output_bpath, speaker)
        for dial_tag in LANG_MAP.keys():
            os.makedirs(os.path.join(opath_speaker, dial_tag), exist_ok=True)

        for idx, row in tqdm(
                unique_test_sentence_df.iterrows(),
                total=len(unique_test_sentence_df),
                desc=f"Speaker {sid}/{len(unique_speakers)}: {speaker}"
        ):
            line = row['sentence']
            if os.path.exists(os.path.join(opath_speaker, dial_tag, f'sent-{idx}.wav')):
                print(f"Skipping {opath_speaker}-{dial_tag}-{idx}")
                continue
            for dial_tag in LANG_MAP.keys():
                tts.tts_to_file(
                    text=line,
                    speaker_wav=conditioning_paths,
                    language=dial_tag,
                    split_sentences=False,
                    file_path=os.path.join(opath_speaker, dial_tag, f'sent-{idx}.wav')
                )


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
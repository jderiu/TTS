import torch
import os, json, argparse
import torchaudio
from huggingface_hub import hf_hub_download
import pandas as pd
from tqdm import tqdm
from ch_test import prepare_unique_sentences, LANG_MAP, LANG_MAP_INV
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def main(config):
    device = "cuda"

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # automatically checks for cached file, optionally set `cache_dir` location
    model_file = hf_hub_download(repo_id='Jenthe/ECAPA2', filename='ecapa2.pt', cache_dir=None)
    ecapa2 = torch.jit.load(model_file, map_location='cuda')
    ecapa2.half()

    model_name = config["model_name"]
    output_bpath = os.path.join(config["output_bpath"], model_name)
    dataspeech_stats_path = config["dataspeech_stats_path"]
    dataspeech_stats_fname = config["dataspeech_stats_fname"]
    test_sentence_path = config["test_sentence_path"]
    test_sentence_fname = config["test_sentnece_fname"]
    n_samples = config["n_samples"]
    speaker_ref_path = config["speaker_ref_path"]

    dataspeech_path = os.path.join(dataspeech_stats_path, dataspeech_stats_fname)
    test_sentence_full_path = os.path.join(test_sentence_path, test_sentence_fname)

    randdf = pd.read_csv(dataspeech_path, sep='\t')
    unique_test_sentence_df = prepare_unique_sentences(test_sentence_full_path, k=n_samples)

    unique_speakers = randdf['speaker_id'].unique()
    output_data = []

    speaker_to_embedding = {}
    speaker_to_avg_similarity = {}
    for sid, speaker in tqdm(enumerate(unique_speakers)):
        speaker_df = randdf[randdf['speaker_id'] == speaker]
        sample_ids = speaker_df['sample_id'].tolist()
        conditioning_paths = [os.path.join(speaker_ref_path, speaker, f"{audio}.wav") for audio in sample_ids]
        ref_embeddings = []
        for cp in conditioning_paths:
            waveform, _ = torchaudio.load(cp)
            embedding = ecapa2(waveform.to(device))
            ref_embeddings.append(embedding.cpu().numpy())

        similarity_matrix = cosine_similarity(np.array(ref_embeddings).squeeze())
        div = len(similarity_matrix) * (len(similarity_matrix) - 1) / 2
        avg_similarity = np.triu(similarity_matrix, k=1).sum() / div
        speaker_to_avg_similarity[speaker] = avg_similarity
        ref_embeddings_avg = np.vstack(ref_embeddings).squeeze().mean(axis=0)
        speaker_to_embedding[speaker] = ref_embeddings_avg

    for sid, speaker in enumerate(unique_speakers):
        opath_speaker = os.path.join(output_bpath, speaker)
        orig_dialect = randdf[randdf['speaker_id'] == speaker]['dialect'].iloc[0]
        orig_dialect_tag = LANG_MAP_INV[orig_dialect]
        ref_embeddings_avg = speaker_to_embedding[speaker]
        for idx, row in tqdm(
                unique_test_sentence_df.iterrows(),
                total=len(unique_test_sentence_df),
                desc=f"Speaker {sid}/{len(unique_speakers)}: {speaker}"
        ):
            for dial_tag in LANG_MAP.keys():
                audio_file = os.path.join(opath_speaker, dial_tag, f'sent-{idx}.wav')
                if not os.path.exists(audio_file):
                    print(f"Skipping {opath_speaker}-{dial_tag}-{idx}")
                    continue
                waveform, _ = torchaudio.load(audio_file)
                sample_embedding = ecapa2(waveform.to(device)).squeeze()
                similarity = float(torch.nn.functional.cosine_similarity(torch.tensor(ref_embeddings_avg, device=device)[None, :], sample_embedding))
                rel_sim = similarity / speaker_to_avg_similarity[speaker]

                output_data.append({
                    'speaker': speaker,
                    "sent_id": idx,
                    'dial_tag': dial_tag,
                    'sentence': row['sentence'],
                    "audio_file": audio_file,
                    'similarity': float(similarity),
                    'rel_sim': float(rel_sim),
                    "orig_dialect_tag": orig_dialect_tag,
                })

    output_df = pd.DataFrame(output_data)
    output_df.to_csv(os.path.join(output_bpath, 'similarity.tsv'), sep='\t', index=False)



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
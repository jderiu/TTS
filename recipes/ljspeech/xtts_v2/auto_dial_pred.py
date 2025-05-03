import os, json, argparse
from transformers import pipeline, AutoTokenizer
import pandas as pd
from ch_test import prepare_unique_sentences
import joblib
import torch
from ch_test import LANG_MAP, LANG_MAP_INV

class_nr_map = {"Bern": 6, "Wallis": 2, "Basel": 5, "Graubünden": 3, "Ostschweiz": 4, "Zürich": 0, "Innerschweiz": 1, "Deutsch": 7}
#class_nr_map = {'Zürich': 0, 'Innerschweiz': 1, 'Wallis': 2, 'Graubünden': 3, 'Ostschweiz': 4, 'Basel': 5, 'Bern': 6}

inverse_cls_map = {v: k for k, v in class_nr_map.items()}

def main(config):
    device = "cuda"

    model_name = config["model_name"]
    output_bpath = os.path.join(config["output_bpath"], model_name)
    dataspeech_stats_path = config["dataspeech_stats_path"]
    dataspeech_stats_fname = config["dataspeech_stats_fname"]
    test_sentence_path = config["test_sentence_path"]
    test_sentence_fname = config["test_sentnece_fname"]
    n_samples = config["n_samples"]
    speaker_ref_path = config["speaker_ref_path"]

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    phoneme_model = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
    eval_batch_size = 32

    try:
        tokenizer = AutoTokenizer.from_pretrained(phoneme_model, force_download=True, trust_remote_code=True)
        print(f"Tokenizer loaded successfully: {tokenizer}")
        print(f"Tokenizer pad_token_id: {tokenizer.pad_token_id}")
    except Exception as e:
        print(f"Tokenizer loading failed: {e}")
        tokenizer = None  # Ensure we don't pass False or None incorrectly

    transcriber = pipeline(model=phoneme_model, device='cuda', batch_size=eval_batch_size, torch_dtype=torch_dtype, tokenizer=tokenizer)
    nb_dial_model = joblib.load('dial_model/text_clf_3_ch_de.joblib')

    dataspeech_path = os.path.join(dataspeech_stats_path, dataspeech_stats_fname)
    test_sentence_full_path = os.path.join(test_sentence_path, test_sentence_fname)

    randdf = pd.read_csv(dataspeech_path, sep='\t')
    unique_test_sentence_df = prepare_unique_sentences(test_sentence_full_path, k=n_samples)

    unique_speakers = randdf['speaker_id'].unique()
    output_data = []
    transcriber.tokenizer = tokenizer
    for sid, speaker in enumerate(unique_speakers):
        opath_speaker = os.path.join(output_bpath, speaker)
        orig_dialect = randdf[randdf['speaker_id'] == speaker]['dialect'].iloc[0]
        orig_dialect_tag = LANG_MAP_INV[orig_dialect]
        for dial_tag in LANG_MAP.keys():
            all_audio_files = []
            for idx, row in unique_test_sentence_df.iterrows():
                audio_file = os.path.join(opath_speaker, dial_tag, f'sent-{idx}.wav')
                if not os.path.exists(audio_file):
                    print(f"Skipping {opath_speaker}-{dial_tag}-{idx}")
                    continue
                all_audio_files.append(audio_file)
            out_strs = transcriber(all_audio_files)
            out_strs = [out_str['text'].replace(' ', '') for out_str in out_strs]

            concat_str = ' '.join(out_strs)
            long_pred = int(nb_dial_model.predict([concat_str])[0])

            # concat batches of 10 transcriptions and predict
            batch_size = 10
            n_batches = len(out_strs) // batch_size
            preds = []
            for i in range(n_batches):
                start = i * batch_size
                end = (i + 1) * batch_size
                batch_cat = ' '.join(out_strs[start:end])
                pred = nb_dial_model.predict([batch_cat])
                preds.extend(pred)

                # max vote over preds
            pred = int(max(set(preds), key=preds.count))
            pred_dial_tag = LANG_MAP_INV[inverse_cls_map[pred]]
            longpred_dial_tag = LANG_MAP_INV[inverse_cls_map[long_pred]]

            output_data.append({
                "speaker_id": speaker,
                "dialect": orig_dialect,
                "dialect_tag": orig_dialect_tag,
                "cond_dialect": dial_tag,
                "pred": pred,
                "pred_tag": pred_dial_tag,
                "long_pred": long_pred,
                "long_pred_tag": longpred_dial_tag
            })

            print(f"Speaker {sid}: {speaker} - {orig_dialect_tag} - {dial_tag} - {pred_dial_tag} - {longpred_dial_tag}")

    output_df = pd.DataFrame(output_data)
    output_df.to_csv(os.path.join(output_bpath, 'dialect_preds.tsv'), sep='\t', index=False)


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
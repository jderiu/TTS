import os, json, argparse

import numpy as np
import pandas as pd

from jiwer import wer
from sacrebleu.metrics import CHRF, TER

from ch_test import prepare_unique_sentences, LANG_MAP
from bleu_metric import score as bleu


chrf = CHRF()
ter = TER()

dialects = LANG_MAP.keys()

def print_results(results, metric):
    print("Results:")
    print(f"{metric}\t", end="")
    for dial in LANG_MAP.keys():
        print(f"{dial}\t", end="")
    print()

    for orig_dial in LANG_MAP.keys():
        print(f"{orig_dial}\t", end="")
        for dial in LANG_MAP.keys():
            r_data = results.get((orig_dial, dial), None)
            if r_data is None:
                continue
            print(f"{r_data[metric]:.4f}\t", end="")
        print()

def main(config):
    model_name = config["model_name"]
    output_bpath = os.path.join(config["output_bpath"], model_name)

    output_df = pd.read_csv(os.path.join(output_bpath, 'back_translation.tsv'), sep='\t')

    dataspeech_stats_path = config["dataspeech_stats_path"]
    dataspeech_stats_fname = config["dataspeech_stats_fname"]
    test_sentence_path = config["test_sentence_path"]
    test_sentence_fname = config["test_sentnece_fname"]
    n_samples = config["n_samples"]

    dataspeech_path = os.path.join(dataspeech_stats_path, dataspeech_stats_fname)
    test_sentence_full_path = os.path.join(test_sentence_path, test_sentence_fname)

    randdf = pd.read_csv(dataspeech_path, sep='\t')
    unique_test_sentence_df = prepare_unique_sentences(test_sentence_full_path, k=n_samples)

    #compute WER, BLEU and CER between the  "sentence" and the "back_translation"

    #lowercase the sentences
    output_df["sentence"] = output_df["sentence"].str.lower()
    output_df["back_translation"] = output_df["back_translation"].str.lower()

    #remove all lines where back_translation is or sentence are not strings
    output_df = output_df[output_df["sentence"].apply(lambda x: isinstance(x, str))]
    output_df = output_df[output_df["back_translation"].apply(lambda x: isinstance(x, str))]

    #sotre results in a dict
    results = {}

    for d1_tag in dialects:
        for d2_tag in dialects:
            #only use those lines where dial_tag equals d2_tag and orig_dial_tag equals d1_tag
            dial_df = output_df[(output_df["orig_dialect_tag"] == d1_tag) & (output_df["dial_tag"] == d2_tag)]

            if len(dial_df) == 0:
                print(f"No data for {d1_tag, d2_tag}")
                continue

            orig_sentences = [[x] for x in dial_df["sentence"].tolist()]
            back_translations = dial_df["back_translation"].tolist()

            bleu_total = bleu(back_translations, [x[0] for x in orig_sentences])
            chrf_total = chrf.corpus_score(back_translations, orig_sentences)
            ter_total = ter.corpus_score(back_translations, orig_sentences)

            wer_scores = []
            for orig, back in zip(orig_sentences, back_translations):
                wer_scores.append(wer(orig, back))

            results[(d1_tag, d2_tag)] = {
                "bleu": bleu_total,
                "chrf": chrf_total.score,
                "ter": ter_total.score,
                "wer": sum(wer_scores)/len(wer_scores)
            }

    #print the results nicely as a table with orig_dial as rows and dial as columns
    print_results(results, "bleu")
    print_results(results, "chrf")
    print_results(results, "ter")
    print_results(results, "wer")

    #compute overall metrics
    orig_sentences = [[x] for x in output_df["sentence"].tolist()]
    back_translations = output_df["back_translation"].tolist()

    bleu_total = bleu(back_translations, [x[0] for x in orig_sentences])
    chrf_total = chrf.corpus_score(back_translations, orig_sentences)
    ter_total = ter.corpus_score(back_translations, orig_sentences)

    wer_scores = []
    for orig, back in zip(orig_sentences, back_translations):
        wer_scores.append(wer(orig, back))

    print("Overall:")
    print(f"BLEU\t{bleu_total}")
    print(f"CHRF\t{chrf_total.score}")
    print(f"TER\t{ter_total.score}")
    print(f"WER\t{sum(wer_scores)/len(wer_scores)}")


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
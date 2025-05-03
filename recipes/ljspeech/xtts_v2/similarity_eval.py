import os, json, argparse

import numpy as np
import pandas as pd

from ch_test import prepare_unique_sentences, LANG_MAP

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

    output_df = pd.read_csv(os.path.join(output_bpath, 'similarity.tsv'), sep='\t')

    results = {}
    for d1_tag in dialects:
        for d2_tag in dialects:
            #only use those lines where dial_tag equals d2_tag and orig_dial_tag equals d1_tag
            dial_df = output_df[(output_df["orig_dialect_tag"] == d1_tag) & (output_df["dial_tag"] == d2_tag)]

            if len(dial_df) == 0:
                print(f"No data for {d1_tag, d2_tag}")
                continue

            similarities = dial_df["similarity"].tolist()
            avg_similarity = sum(similarities)/len(similarities)

            rel_similarities = dial_df["rel_sim"].tolist()
            avg_rel_similarity = sum(rel_similarities)/len(rel_similarities)

            results[(d1_tag, d2_tag)] = {
                "avg_similarity": avg_similarity,
                "avg_rel_similarity": avg_rel_similarity
            }

    print_results(results, "avg_similarity")
    print_results(results, "avg_rel_similarity")

    #compute overall metrics
    similarities = output_df["similarity"].tolist()
    rel_similarities = output_df["rel_sim"].tolist()
    avg_similarity = sum(similarities)/len(similarities)
    avg_rel_similarity = sum(rel_similarities)/len(rel_similarities)

    print("Overall:")
    print(f"avg_similarity: {avg_similarity:.4f}")
    print(f"avg_rel_similarity: {avg_rel_similarity:.4f}")


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
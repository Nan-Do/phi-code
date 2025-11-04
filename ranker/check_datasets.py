import argparse
import glob
import json
import os
import sys

from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def main(args):
    print("Loading model")
    model = SentenceTransformer(args.model, trust_remote_code=True)
    tokenizer = model.tokenizer

    if args.limit > tokenizer.model_max_length:
        print(
            f"WARNING: The specified max length({args.limit}) is higher than the one supported by the tokenizer({tokenizer.model_max_length})"
        )
        limit = tokenizer.model_max_length
    else:
        limit = args.limit

    if not args.limit:
        limit = tokenizer.model_max_length

    print(f"Using limit: {limit}")

    fpaths = []
    if args.dataset == "all":
        fpaths = list(glob.glob("*.jsonl"))
    else:
        fpaths = [args.dataset]

    for fpath in fpaths:
        max_length, good, bad = 0, 0, 0
        max_length_tokens = 0
        print(f"Processing {fpath}")
        with open(fpath) as r:
            for line in tqdm(r):
                show_values = False
                pos_tokens, neg_tokens, anchor_tokens = 0, 0, 0
                obj = json.loads(line)
                for key in obj:
                    try:
                        text = obj[key]
                        if not text:
                            raise TypeError

                        encoded_text = tokenizer.encode(text)
                        max_length = max(max_length, len(text))
                        max_length_tokens = max(max_length_tokens, len(encoded_text))
                        if key == "anchor":
                            anchor_tokens = len(encoded_text)
                        elif key == "positive":
                            pos_tokens = len(encoded_text)
                        else:
                            neg_tokens = len(encoded_text)
                    except TypeError:
                        print(f"ERROR with sample {obj}")
                        sys.exit(1)

                if ((anchor_tokens + pos_tokens) > limit) or (
                    (anchor_tokens + neg_tokens) > limit
                ):
                    bad += 1
                else:
                    good += 1

        print(
            f"Max Length: {max_length}, Max Length Tokens: {max_length_tokens}, Good samples: {good}, Bad samples: {bad}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="check the integrity the given datasets (including required fields and tokenized length)"
    )

    parser.add_argument(
        "--model",
        "-m",
        default="coldchair16/CPRetriever-Code",
        type=str,
        help="name of the model to fine-tune (by default coldchair16/CPRetriever-Code)",
    )

    parser.add_argument(
        "--dataset",
        "-d",
        default="all",
        type=str,
        help="specify the jsonl file containing the dataset (all will check all the jsonl files in the current directory)",
    )

    parser.add_argument(
        "--limit",
        "-l",
        default=0,
        type=int,
        help="specify the limit of the tokenizer (by default it will use the tokenizer's max length)",
    )

    args = parser.parse_args()
    main(args)

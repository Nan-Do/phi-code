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

    fpaths = []
    if args.dataset == "all":
        fpaths = list(glob.glob("*.jsonl"))
    else:
        fpaths = [args.dataset]

    for fpath in fpaths:
        max_length, good, bad = 0, 0, 0
        max_length_tokens = 0
        fname = os.path.basename(fpath).split(".")[0]
        print(f"Processing {fpath}")
        with open(fpath) as r:
            with open(f"{fname}-filtered.jsonl", "w") as w:
                pos_tokens, neg_tokens, anchor_tokens = 0, 0, 0
                for line in tqdm(r):
                    obj = json.loads(line)
                    for key in obj:
                        try:
                            text = obj[key]
                            encoded_text = tokenizer.encode(text)
                            max_length = max(max_length, len(text))
                            max_length_tokens = max(
                                max_length_tokens, len(encoded_text)
                            )
                            if key == "anchor":
                                anchor_tokens = len(encoded_text)
                            elif key == "positive":
                                pos_tokens = len(encoded_text)
                            else:
                                neg_tokens = len(encoded_text)
                        except TypeError:
                            print(f"Error with object {obj}")
                            sys.exit(1)

                    if ((anchor_tokens + pos_tokens) > limit) or (
                        (anchor_tokens + neg_tokens) > limit
                    ):
                        bad += 1
                    else:
                        w.write(line)
                        good += 1

        print(f"Max Length: {max_length}, Good samples: {good}, Bad samples: {bad}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="filter the datasets removing the samples that go over the tokenizer limit"
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

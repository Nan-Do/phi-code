import argparse
import json
import random
from tqdm import tqdm


def main(args):
    file = args.file
    percentage = args.percentage

    print(f"Loading {file}")
    data = []
    with open(file) as f:
        for line in tqdm(f.readlines()):
            data.append(json.loads(line))

    random.shuffle(data)
    to_save = (len(data) * percentage) // 100
    print(f"Storing {to_save} samples")
    save_file = file[:-6]
    with open(save_file + "_small.jsonl", "w") as f:
        for i in tqdm(range(to_save)):
            f.write(json.dumps(data[i]))
            f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="sample a given percentage of the total samples of the dataset"
    )

    parser.add_argument(
        "--file",
        "-f",
        required=True,
        type=str,
        help="path to the dataset in jsonl format",
    )

    parser.add_argument(
        "--percentage",
        "-p",
        default=10,
        choices=[10, 20, 30, 40, 50, 60, 70, 80, 90],
        type=int,
        help="percentage of samples (10 by default)",
    )

    args = parser.parse_args()
    main(args)

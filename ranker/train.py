import argparse
import json
import torch

from collections import defaultdict
from datasets import Dataset
from log_config import log
from random import shuffle
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import TripletLoss, CachedMultipleNegativesRankingLoss
from sentence_transformers.losses import TripletDistanceMetric
from sentence_transformers.training_args import MultiDatasetBatchSamplers, BatchSamplers


class Range(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __eq__(self, other):
        return self.start <= other <= self.end

    def __str__(self):
        return f"[{self.start}, {self.end}]"


def parse_datasets(args):
    datasets = []
    if args.leetcode:
        datasets.append(("Leetcode", args.leetcode))
    if args.atcoder:
        datasets.append(("Atcoder", args.atcoder))
    if args.codechef:
        datasets.append(("Codechef", args.codechef))
    if args.codeforces:
        datasets.append(("Codeforces", args.codeforces))
    if args.codeforces_positive:
        datasets.append(("CodeforcesPositive", args.codeforces_positive))

    return datasets


def process_data(data):
    processed_data = defaultdict(list)
    for elem in data:
        for key in elem:
            processed_data[key].append(elem[key])
    return processed_data


def load_dataset(dataset_path, eval_size=0.1):
    data = []
    with open(dataset_path) as f:
        for line in f.readlines():
            data.append(json.loads(line))
    shuffle(data)
    train_size = int(len(data) * (1 - eval_size))
    train_data = process_data(data[:train_size])
    eval_data = process_data(data[train_size:])
    return train_data, eval_data


def main(args):
    datasets = parse_datasets(args)
    log.info("Loading datasets")
    if not datasets:
        log.error("You must include at least one dataset!")
        return

    train, eval = {}, {}
    for name, dataset_path in datasets:
        log.info(f"Loading {name} dataset")
        train_data, eval_data = load_dataset(dataset_path, args.eval_size)
        train[name] = Dataset.from_dict(train_data)
        eval[name] = Dataset.from_dict(eval_data)

    model_kwargs = {
        "torch_dtype": torch.bfloat16
    }  # TODO: Make sure your GPU and model support bfloat16
    log.info(f"Loading the model: {args.model}")
    model = SentenceTransformer(
        args.model, trust_remote_code=True, model_kwargs=model_kwargs
    )

    losses = {}
    for name in train.keys():
        if name in ["Leetcode", "CodeforcesPositive"]:
            losses[name] = CachedMultipleNegativesRankingLoss(
                model, mini_batch_size=args.mini_batch_size, scale=20
            )
        else:
            losses[name] = TripletLoss(
                model,
                TripletDistanceMetric.COSINE,
                triplet_margin=args.triplet_margin,
            )

    args = SentenceTransformerTrainingArguments(
        output_dir=args.output_dir,
        gradient_accumulation_steps=args.accumulation_steps,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        lr_scheduler_type="constant",
        bf16=True,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=args.logging_steps,
        run_name="Ranker",
        save_on_each_node=False,
        max_grad_norm=1.0,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        multi_dataset_batch_sampler=MultiDatasetBatchSamplers.PROPORTIONAL,
    )

    trainer = SentenceTransformerTrainer(
        model=model, train_dataset=train, args=args, loss=losses, eval_dataset=eval
    )
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="fine-tuner for Delta-Code's ranker agent"
    )

    parser.add_argument(
        "--model",
        "-m",
        default="coldchair16/CPRetriever-Code",
        type=str,
        help="name of the model to fine-tune (by default coldchair16/CPRetriever-Code)",
    )

    parser.add_argument(
        "--eval_size",
        "-s",
        default=0.1,
        type=float,
        choices=[Range(0.0, 1.0)],
        help="percentage of the dataset used for evaluation (default 0.1)",
    )

    parser.add_argument(
        "--eval_steps",
        "-p",
        default=100000,
        type=int,
        help="how many steps to wait before running the evaluation",
    )

    parser.add_argument(
        "--epochs",
        "-e",
        default=1,
        type=int,
        help="number of epochs to train (1 by default)",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        default="working_dir",
        type=str,
        help="where to store the generated data (working_dir by default)",
    )

    parser.add_argument(
        "--logging-steps",
        "-l",
        default=1000,
        type=int,
        help="logging steps (1000 by default)",
    )

    parser.add_argument(
        "--accumulation-steps",
        "-a",
        default=1,
        type=int,
        help="gradient accumulation steps (1 by default)",
    )

    parser.add_argument(
        "--batch-size",
        "-b",
        default=1,
        type=int,
        help="training batch size (1 by default)",
    )

    parser.add_argument(
        "--mini-batch-size",
        "-t",
        default=2,
        type=int,
        help="training mini batch size for the Multiple Negative Loss (2 by default)",
    )

    parser.add_argument("--leetcode", type=str, help="path to the leetcode dataset")

    parser.add_argument("--atcoder", type=str, help="path to the atcoder dataset")

    parser.add_argument("--codeforces", type=str, help="path to the codeforces dataset")

    parser.add_argument(
        "--codeforces_positive",
        type=str,
        help="path to the codeforces with positive samples dataset",
    )

    parser.add_argument("--codechef", type=str, help="path to the codechef dataset")

    parser.add_argument(
        "--triplet_margin", default=0.3, type=float, help="The margin for triplet loss"
    )

    args = parser.parse_args()
    main(args)

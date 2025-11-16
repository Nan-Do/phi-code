import argparse
import os
import torch
import sys

from leetcode import get_leetcode_problem_tests, run_leetcode_tests
from log_config import log
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from web_ui import build_web_ui
from terminal import run_terminal_mode


def check_positive(value):
    try:
        value = int(value)
        if value <= 0:
            raise argparse.ArgumentTypeError(f"{value} is not a positive integer")
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} is not a positive integer")
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="φ-Code: A Python Agentic Competitive Programmer Fueled by (tiny) LLMs."
    )

    parser.add_argument(
        "-r",
        "--ranker",
        metavar="ranker_model",
        help="path to the model that will be used as a ranker. It can be a HuggingFace link (Salesforce/SFR-Embedding-Code-2B_R).",
        default="Salesforce/SFR-Embedding-Code-2B_R",
        type=str,
    )

    parser.add_argument(
        "-s",
        "--server",
        metavar="server",
        help="address of the server running a llama.cpp server.",
        required=True,
        type=str,
    )

    parser.add_argument(
        "-p",
        "--port",
        metavar="port",
        help="port of the server running a llama.cpp server.",
        required=True,
        type=str,
    )

    parser.add_argument(
        "-m",
        "--site",
        metavar="site",
        help="specify the competitive web site the problems are coming from (Options: leetcode).",
        required=True,
        choices=["leetcode"],
        type=str,
    )

    parser.add_argument(
        "-i",
        "--interface",
        metavar="interface",
        help="specify the interface for the app (Options: web, terminal).",
        choices=["web", "terminal"],
        default="web",
        type=str,
    )

    parser.add_argument(
        "-f",
        "--statement",
        metavar="statement",
        help="specify a file containing the problem statement to solve.",
        type=str,
    )

    parser.add_argument(
        "-n",
        "--number",
        metavar="number",
        help="specify the number of solutions to generate.",
        type=check_positive,
    )

    parser.add_argument(
        "-o",
        "--output_file",
        metavar="output_file",
        help="specify the output file to store the results in jsonl format (if the file exists it will be overwritten).",
        type=str,
    )

    args = parser.parse_args()
    ranker = args.ranker
    server = args.server
    port = args.port
    site = args.site
    interface = args.interface
    num_solutions = args.number
    statement_file = args.statement
    output_file = args.output_file

    # Check that the provided statment file exists and read it.
    statement = None
    if not os.path.exists(statement_file):
        log.error("The provided statement file doesn't exist.")
        if interface == "terminal":
            log.error(
                "For the terminal interface to work the statement file must exist."
            )
            sys.exit(1)
    else:
        with open(statement_file) as f:
            statement = f.read()

    if interface == "terminal" and output_file is None:
        log.error("For the terminal interface to work you must specify an output file.")
        sys.exit(1)

    # Don't show the warning about forking causing problems the tokenizer is
    # only used in the main thread.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if site == "leetcode":
        prompt_file = "./leetcode_prompt.txt"
        get_problem_tests = get_leetcode_problem_tests
        run_tests = run_leetcode_tests
    else:
        log.error("The specified site is unknown, please use a valid site.")
        sys.exit(1)

    log.info("Reading the prompt file.")
    with open(prompt_file) as f:
        prompt_template = f.read()

    log.info("Creating the client for the llama.cpp server.")
    client = OpenAI(base_url=f"http://{server}:{port}", api_key="dummy-key")

    log.info(f"Loading the ranker model {args.ranker}.")
    model_kwargs = {"torch_dtype": torch.bfloat16}
    ranker = SentenceTransformer(
        args.ranker, trust_remote_code=True, model_kwargs=model_kwargs
    )
    log.info("Done")

    if interface == "web":
        log.info("Using the web interface.")
        app = build_web_ui(
            prompt_template,
            client,
            ranker,
            get_problem_tests,
            run_tests,
            num_solutions,
            statement,
        )
        app.launch()
    elif interface == "terminal":
        log.info("Running the app in terminal mode, no interface will be used.")
        run_terminal_mode(
            prompt_template,
            client,
            ranker,
            get_problem_tests,
            run_tests,
            num_solutions,
            statement,
            output_file,
        )

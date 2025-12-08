from log_config import log
from multiprocessing import Process, Queue
from tqdm import tqdm
from typing import List, Tuple, Dict
from utils import DEBUG

import tempfile
import time
import subprocess


def process_sample(curr_line: int, lines: List[str]) -> Tuple[int, list[str]]:
    data = []
    curr_line += 1
    if lines[curr_line].strip().startswith("Copy"):
        curr_line += 1
    if not lines[curr_line].strip():
        curr_line += 1
    while (
        curr_line < len(lines)
        and lines[curr_line].strip()
        and not lines[curr_line].strip().startswith("Input")
        and not lines[curr_line].strip().startswith("Output")
    ):
        data.append(lines[curr_line][:])
        curr_line += 1
    return curr_line, data


def get_codeforces_problems_tests(problem_statement: str):
    """
    Get the problem tests from an atcoder problem statement and format them into a JSON object
    """
    log.info("Obtaining the problem tests.")

    # Filter the input and output samples from the problem statement
    curr_line = 0
    input_samples, output_samples = [], []
    lines = problem_statement.split("\n")

    inputs_seen = 0
    total_inputs = len(list(filter(lambda x: x.startswith("Input"), lines)))
    if total_inputs == 0:
        log.error("Problem examples not found.")
        return []

    while curr_line < len(lines):
        if lines[curr_line].strip().startswith("Input"):
            inputs_seen += 1
            if (total_inputs == 2) and (inputs_seen < 2):
                curr_line += 1
                continue
            curr_line, input_samples = process_sample(curr_line, lines)

        if lines[curr_line].strip().startswith("Output"):
            if (total_inputs == 2) and (inputs_seen < 2):
                curr_line += 1
                continue
            curr_line, output_samples = process_sample(curr_line, lines)

        curr_line += 1

    log.info(f"{len(output_samples)} tests obtained in 1 sample")

    problem_tests = [
        {"input": "\n".join(input_samples), "output": "\n".join(output_samples)}
    ]

    return problem_tests


def run_codeforces_test(
    solution_position: int,
    code: str,
    problem_tests: List[Dict[str, str]],
    procs_queue: Queue,
):
    if DEBUG:
        log.info(f"Code: {code}")

    # Create a temporary directory to run the code and store the results.
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Store the code
        with open(f"{tmpdirname}/code.py", "w") as code_file:
            code_file.write(code)

        # Check the tests
        tests_passed = 0
        for i, problem_test in enumerate(problem_tests):
            process = subprocess.Popen(
                [
                    "python3",
                    f"{tmpdirname}/code.py",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            stdout_data, _ = process.communicate(problem_test["input"])

            if not stdout_data:
                log.error("No ouput generated for the generated code, skipping.")
                break

            if stdout_data.strip() == problem_test["output"]:
                tests_passed += 1

    procs_queue.put((solution_position, tests_passed))


def run_codeforces_tests(
    generated_solutions, problem_tests, timeout_seconds=2, num_processes=1
):
    code_queue = []
    # First initialize the tests passed for each solution to 0.
    for pos, generated_solution in enumerate(generated_solutions):
        generated_solution["tests_passed"] = 0
        code_queue.append((pos, generated_solution["code"]))

    if not problem_tests:
        log.error(
            "No valid problem tests were found, check the problem statement. Skipping tests"
        )
        return

    # Initialize the progress bar for the problem tests
    pbar = tqdm(total=len(code_queue))
    procs_queue = Queue()
    while code_queue:
        procs = []
        while code_queue and len(procs) < num_processes:
            solution_position, code = code_queue.pop()
            p = Process(
                target=run_codeforces_test,
                args=(solution_position, code, problem_tests, procs_queue),
            )
            procs.append(p)

        for p in procs:
            p.start()

        start = time.time()
        while time.time() - start <= timeout_seconds:
            if not any(p.is_alive() for p in procs):
                break
            time.sleep(0.1)
        else:
            log.error("Test execution timedout, killing all processes")
            for p in procs:
                p.terminate()
                p.join()

        # At this point all tests have finished.
        # Update the progress bar with the finished tests
        pbar.update(len(procs))

        while not procs_queue.empty():
            solution_position, num_tests_passed = procs_queue.get()
            generated_solutions[solution_position]["tests_passed"] = num_tests_passed

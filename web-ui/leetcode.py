import ast
import time
import re

from log_config import log
from multiprocessing import Process, Queue
from tqdm import tqdm
from typing import List
from utils import DEBUG

regex = re.compile(r", *(?=[a-zA-Z_][a-zA-Z0-9_]* *=)")


def get_leetcode_problem_tests(problem_statement: str):
    """
    Get the problem tests from a leetcode problem statement and format them into a JSON object
    """
    log.info("Obtaining the problem tests.")
    # Filter the input and output fields from the problem statement
    input_fields = list(
        filter(lambda x: x.strip().startswith("Input:"), problem_statement.split("\n"))
    )
    output_fields = list(
        filter(lambda x: x.strip().startswith("Output:"), problem_statement.split("\n"))
    )
    if len(input_fields) != len(output_fields):
        log.error(
            "The number of input and output tests doesn't match for the given problem statement"
        )
        min_len = min(len(input_fields), len(output_fields))
        input_fields = input_fields[:min_len]
        output_fields = output_fields[:min_len]

    log.info(f"{len(input_fields)} tests obtained")

    # For each input: "colors=3, times=[1,2,3,4]"
    # Parse it using regexp into a dictionary as {'colors': 3, 'times': [1,2,3,4]}
    # Same for output but ouput is always a single element.
    problem_tests = []
    for i in range(len(input_fields)):
        input_data = {}
        for pair in re.split(regex, input_fields[i][7:].strip()):
            if "=" not in pair:
                continue

            var_name, value_str = pair.split("=", 1)
            var_name = var_name.strip()
            value_str = value_str.strip()

            try:
                parsed_value = ast.literal_eval(value_str)
            except (ValueError, SyntaxError) as e:
                log.error(
                    f"Error parsing value for '{var_name}' ('{value_str}'): {e}. Skipping."
                )
                continue
            input_data[var_name] = parsed_value
        if not input_data:
            log.error(f"Input data not found for example {input_fields[i]}, Skipping")
        try:
            problem_tests.append(
                {
                    "input": input_data,
                    "output": ast.literal_eval(output_fields[i][7:].strip()),
                }
            )
        except (ValueError, SyntaxError) as e:
            log.error(
                f"Error parsing value for 'Output' ('{output_fields[i][7:].strip()}'): {e}. Skipping."
            )
            continue

    return problem_tests


def run_leetcode_test(
    solution_position: int, code: str, problem_tests: List[str], procs_queue: Queue
):
    # Import some common types as sometimes the models forget.
    from typing import List, Dict, Set, Union, Tuple, Sequence

    if DEBUG:
        log.info(f"Code: {code}")

    # Store the namespace
    ns = {}

    # Get the solution class and get the function name used for the problem
    try:
        exec(code, locals(), ns)
    except SyntaxError:
        log.error("Generated solution contains malformed code. Skipping")
        return

    if "Solution" not in ns:
        log.error("Generated solution didn't create a Solution class. Skipping")
        return

    # If the generated code imports some symbol we need to import it here too
    for val in ns:
        if val == "Solution":
            solution_class = ns["Solution"]()
        else:
            locals()[val] = ns[val]
    func_name = [
        name
        for name in dir(solution_class)
        if callable(getattr(solution_class, name)) and not name.startswith("__")
    ][0]
    if DEBUG:
        log.info(f"Generated function {func_name}")
    func = getattr(solution_class, func_name)

    # Execute the tests with the generated code
    tests_passed = 0
    for example in problem_tests:
        try:
            if func(**example["input"]) == example["output"]:
                tests_passed += 1
        except (TypeError, IndexError, NameError) as e:
            log.error(f'"{e}": calling the generated solution function. Skipping')

    procs_queue.put((solution_position, tests_passed))


def run_leetcode_tests(
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
                target=run_leetcode_test,
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

    pbar.close()

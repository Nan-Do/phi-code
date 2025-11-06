import ast
import re

from log_config import log
from tqdm import tqdm
from utils import DEBUG

regex = re.compile(r", *(?=[a-zA-Z_][a-zA-Z0-9_]* *=)")


def get_leetcode_problem_tests(problem_statement: str):
    """
    Get the problem tests from a leetcode problem statement and format them into a JSON object
    """
    # --- Get the problem tests from Leetcode other sites won't work ---
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
        return None

    # For each input: "colors=3, times=[1,2,3,4]"
    # Parse it using regexp into a dictionary as {'colors': 3, 'times': [1,2,3,4]}
    # Same for output.
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


def run_leetcode_tests(generated_solutions, problem_tests):
    for generated_solution in tqdm(generated_solutions):
        generated_solution["tests_passed"] = 0

        if DEBUG:
            log.info(f"Code: {generated_solution['code']}")
        ns = {}

        # Get the solution class and get the function name used for the problem
        try:
            exec(generated_solution["code"], globals(), ns)
        except SyntaxError:
            log.error("Generated solution contains malformed code. Skipping")
            continue

        if "Solution" not in ns:
            log.error("Generated solution didn't create a Solution class. Skipping")
            continue

        # If the generated code imports some symbol we need to import it here too
        for val in ns:
            if val == "Solution":
                solution_class = ns["Solution"]()
            else:
                globals()[val] = ns[val]
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
            except TypeError:
                log.error(
                    "There was an error calling the generated solution function. Skipping"
                )
        generated_solution["tests_passed"] = tests_passed

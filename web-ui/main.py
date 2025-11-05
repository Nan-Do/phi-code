import argparse
import ast
import gradio as gr
import re
import torch

from log_config import log
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DEBUG = False

regex = re.compile(r", *(?=[a-zA-Z_][a-zA-Z0-9_]* *=)")


def get_problem_tests(problem_statement: str):
    """
    Get the problem tests from the problem statement and format them into a JSON object
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


def generate_solutions(num_solutions: int, problem_statement: str):
    """
    Fills the prompt, queries the llama server, generates num_solutions,
    ranks them and prepares the UI for display.
    """
    # --- Parse the problem tests ---
    problem_tests = get_problem_tests(problem_statement)
    if DEBUG:
        log.info(f"Problem tests: {problem_tests}")

    # --- Generate the prompt for the LLM ---
    prompt = prompt_template.format(problem_statement)
    if DEBUG:
        log.info(f"Prompt: {prompt}")

    # --- Generate the solutions ---
    code_generations = []
    generated_solutions = []
    log.info(f"Generating {num_solutions} solutions:")
    for number in tqdm(range(num_solutions)):
        try:
            # Get the completions only non reasoning completions are supported
            # If the completion goes over the limit of tokens or doesn't finish
            # in a proper way skip it.
            completion = client.chat.completions.create(
                model="coder-model", messages=[{"role": "user", "content": prompt}]
            )
            choice = completion.choices[0]
            if choice.finish_reason != "stop":
                log.error(
                    f"Solution generation {number} finished with {choice.finish_reason} instead of a stop. Skipping it."
                )
            # Strip the markdown in case the model used it in its answer.
            code_body: str = choice.message.content
            if code_body.startswith("```"):
                code_body = "\n".join(code_body.split("\n")[1:-1])
            # Add the code to the solutions, one will be used for ranking and the other for the gradio ui
            code_generations.append(code_body)
            generated_solutions.append({"code": code_body})
        except Exception as e:
            log.error(f"An error occurred generating a solution: {e}")

    # --- Rank the generated solutions ---
    log.info("Ranking the solutions")
    anchor_embedding = ranker.encode(problem_statement, convert_to_tensor=True)
    generation_embeddings = ranker.encode(code_generations, convert_to_tensor=True)
    scores_tensors = ranker.similarity(anchor_embedding, generation_embeddings)
    for pos, score in enumerate(scores_tensors.squeeze().tolist()):
        generated_solutions[pos]["score"] = score

    if DEBUG:
        print(f"Scores: {scores_tensors.squeeze().tolist()}")

    # --- Compute the number of passed tests ---
    log.info("Computing the number of passed tests")
    for pos, solution in enumerate(generated_solutions):
        generated_solutions[pos]["tests_passed"] = 0

        if DEBUG:
            log.info(f"Code: {solution['code']}")
        ns = {}

        # Get the solution class and get the function name used for the problem
        try:
            exec(solution["code"], globals(), ns)
        except SyntaxError:
            log.error("Generated solution contains malformed code. Skipping")
            continue

        # If the genrated code imports some symbol we need to import it here too
        for val in ns:
            if val == "Solution":
                solution = ns["Solution"]()
            else:
                globals()[val] = ns[val]
        func_name = [
            name
            for name in dir(ns["Solution"])
            if callable(getattr(ns["Solution"], name)) and not name.startswith("__")
        ][0]
        if DEBUG:
            log.info(f"Generated function {func_name}")
        func = getattr(solution, func_name)

        # Execute the tests with the generated code
        tests_passed = 0
        for example in problem_tests:
            if func(**example["input"]) == example["output"]:
                tests_passed += 1
        generated_solutions[pos]["tests_passed"] = tests_passed
        if DEBUG:
            log.info(f"Solution {pos}: {tests_passed}/{len(problem_tests)}")

    # Sort the solutions by the number of tests passed and the ranking score
    generated_solutions.sort(
        key=lambda dict: (dict.get("tests_passed"), dict.get("score")), reverse=True
    )

    # After generating, get the first solution to display immediately
    initial_index, initial_code, initial_score, initial_status, initial_tests = (
        change_solution(generated_solutions, -1, "Next")
    )

    # Return all the values needed to update the Gradio components,
    # including making the hidden components visible.
    return (
        generated_solutions,  # Update solutions_state
        initial_index,  # Update current_index_state to 0
        initial_code,  # Update the code block
        initial_score,  # Update the score label
        initial_status,  # Update the status label
        initial_tests,  # Update the number of tests passed
        gr.update(visible=True),  # Make the solution display group visible
        gr.update(visible=True),  # Make the navigation buttons visible
    )


def change_solution(solutions, current_index, direction):
    """
    Updates the displayed solution based on the navigation direction.

    Args:
        solutions (list): The list of all solution dictionaries.
        current_index (int): The index of the currently displayed solution.
        direction (str): Either "Previous" or "Next".

    Returns:
        tuple: A tuple containing the new index, the code for the new solution,
               the score, and the status text.
    """
    total_solutions = len(solutions)

    # Gracefully handle the case where there are no solutions.
    if total_solutions == 0:
        return 0, "No code to display.", 0.0, "No solutions generated."

    if direction == "Next":
        # Move to the next index, but don't go past the last one
        new_index = min(current_index + 1, total_solutions - 1)
    elif direction == "Previous":
        # Move to the previous index, but don't go below the first one
        new_index = max(current_index - 1, 0)
    else:
        # This case is used for the initial load after generation
        new_index = current_index

    # Get the new solution to display
    solution = solutions[new_index]

    # Format the status text
    status = f"Solution {new_index + 1} of {total_solutions}"
    tests = f"Passed: {solution['tests_passed']}"

    return new_index, solution["code"], solution["score"], status, tests


# --- Gradio UI ---

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # φ-Code: A Python Agentic Competitive Programmer Fueled by (tiny) LLMs.
        Click the generate button to get solutions from the model, then use the buttons to browse through them.
        """
    )

    problem_statement = gr.Textbox(label="Problem Statement:", lines=45)
    num_solutions = gr.Number(label="Number of Solutions to Generate:", value=10)

    # State variables to hold the solutions and the current index
    solutions_state = gr.State(value=[])
    current_index_state = gr.State(value=0)

    # --- UI Components ---
    generate_btn = gr.Button("🚀 Generate Solutions", variant="primary")

    # Group for solution display, initially hidden
    with gr.Group(visible=False) as solution_display_group:
        with gr.Row():
            solution_status = gr.Label(label="Status", scale=1)
            solution_tests = gr.Label(label="Tests", scale=1)
            solution_score = gr.Label(label="Confidence Score", scale=1)

        solution_code = gr.Code(language="python", label="Code Solution")

    # Row for navigation buttons, also initially hidden
    with gr.Row(visible=False) as nav_buttons_row:
        prev_btn = gr.Button("⬅️ Previous Solution")
        next_btn = gr.Button("Next Solution ➡️")

    # --- Event Handling ---

    # When the "Generate" button is clicked, call the generation function
    # and update all the necessary UI components.
    generate_btn.click(
        fn=generate_solutions,
        inputs=[num_solutions, problem_statement],
        outputs=[
            solutions_state,
            current_index_state,
            solution_code,
            solution_score,
            solution_status,
            solution_tests,
            solution_display_group,
            nav_buttons_row,
        ],
    )

    # When the "Previous" button is clicked
    prev_btn.click(
        fn=change_solution,
        inputs=[
            solutions_state,
            current_index_state,
            gr.Textbox("Previous", visible=False),  # Pass "Previous" as direction
        ],
        outputs=[
            current_index_state,
            solution_code,
            solution_score,
            solution_status,
            solution_tests,
        ],
    )

    # When the "Next" button is clicked
    next_btn.click(
        fn=change_solution,
        inputs=[
            solutions_state,
            current_index_state,
            gr.Textbox("Next", visible=False),  # Pass "Next" as direction
        ],
        outputs=[
            current_index_state,
            solution_code,
            solution_score,
            solution_status,
            solution_tests,
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="φ-Code: A Python Agentic Competitive Programmer Fueled by (tiny) LLMs."
    )

    parser.add_argument(
        "-r",
        "--ranker",
        metavar="ranker_model",
        help="path to the model that will be used as a ranker. It can be a HuggingFace link (Salesforce/SFR-Embedding-Code-2B_R)",
        default="Salesforce/SFR-Embedding-Code-2B_R",
        type=str,
    )

    parser.add_argument(
        "-s",
        "--server",
        metavar="server",
        help="address of the server running a llama.cpp server",
        required=True,
        type=str,
    )

    parser.add_argument(
        "-p",
        "--port",
        metavar="port",
        help="port of the server running a llama.cpp server",
        required=True,
        type=str,
    )

    parser.add_argument(
        "-t",
        "--prompt",
        metavar="prompt",
        help="file containing the prompt template for the LLM.",
        default="prompt.txt",
        type=str,
    )

    args = parser.parse_args()
    ranker = args.ranker
    server = args.server
    port = args.port
    prompt_file = args.prompt

    log.info("Reading the prompt file")
    with open(prompt_file) as f:
        prompt_template = f.read()

    log.info("Creating the client for the llama.cpp server")
    client = OpenAI(base_url=f"http://{server}:{port}", api_key="dummy-key")

    log.info(f"Loading the ranker model {args.ranker}")
    model_kwargs = {"torch_dtype": torch.bfloat16}
    ranker = SentenceTransformer(
        args.ranker, trust_remote_code=True, model_kwargs=model_kwargs
    )
    log.info("Done")

    demo.launch()

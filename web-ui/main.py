import argparse
import gradio as gr
import os
import torch
import sys

from leetcode import get_leetcode_problem_tests, run_leetcode_tests
from log_config import log
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from utils import DEBUG, get_completions_from_prompt


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
    run_analysis = True
    log.info(f"Generating {num_solutions} solutions:")
    for _ in tqdm(range(num_solutions)):
        code_generation = get_completions_from_prompt(client, prompt, 2)
        if code_generation:
            code_generations.append(code_generation)
            generated_solutions.append({"code": code_generation})

    # The model couldn't generate any valid solution. Update the
    # solutions to notify the user in the web ui, skip the analysis
    # and log the error.
    if not generated_solutions:
        run_analysis = False
        log.error(
            "No valid solution was generated, skipping checks, please generate new solutions."
        )
        generated_solutions = [
            {
                "code": "THERE HAS BEEN A PROBLEM WITH THE GENERATION PROCESS\nPLEASE PRESS THE GENERATE SOLUTION BUTTONS AGAIN.",
                "tests_passed": 0,
                "score": 0.0,
            }
        ]

    # If everthing has been ok so far, check how many tests each solution passes and rank them.
    if run_analysis:
        # --- Compute the number of passed tests ---
        log.info("Running tests:")
        run_tests(generated_solutions, problem_tests)

        # --- Rank the generated solutions ---
        log.info("Ranking the solutions.")
        anchor_embedding = ranker.encode(problem_statement, convert_to_tensor=True)
        generation_embeddings = ranker.encode(code_generations, convert_to_tensor=True)
        scores_tensors = ranker.similarity(anchor_embedding, generation_embeddings)

        if DEBUG:
            print(f"Scores: {scores_tensors.squeeze().tolist()}")

        # tolist() doesn't return a list of 1 so if there is one solution
        # store the value appropriately.
        if len(generated_solutions) > 1:
            for pos, score in enumerate(scores_tensors.squeeze().tolist()):
                generated_solutions[pos]["score"] = score
        else:
            generated_solutions[0]["score"] = scores_tensors.squeeze().tolist()

    # Sort the solutions by the number of tests passed and the ranking score
    generated_solutions.sort(
        key=lambda dict: (dict.get("tests_passed"), dict.get("score")), reverse=True
    )

    # After generating, get the first solution to display immediately
    initial_index, initial_code, initial_score, initial_status, initial_tests = (
        change_solution(generated_solutions, -1, "Next")
    )

    # Log that the process has ended.
    if run_analysis:
        add_s = "s" if len(generated_solutions) != 1 else ""
        log.info(
            f"Process finished, {len(generated_solutions)} solution{add_s} generated"
        )
    else:
        log.info(
            f"Process finished, no solutions were generated, please generate new solutions."
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
        "-m",
        "--site",
        metavar="site",
        help="specify the site for the tool to work (Options: ['leetcode']",
        required=True,
        choices=["leetcode"],
        type=str,
    )

    args = parser.parse_args()
    ranker = args.ranker
    server = args.server
    port = args.port
    site = args.site

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

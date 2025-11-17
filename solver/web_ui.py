import gradio as gr

from utils import DEBUG, get_completions_from_prompt
from terminal import run_terminal_mode

prompt_template = ""
client, ranker = None, None
get_problem_tests, run_tests = None, None


def generate_solutions(num_solutions: int, problem_statement: str):
    """
    Fills the prompt, queries the llama server, generates num_solutions,
    ranks them and prepares the UI for display.
    """
    generated_solutions = run_terminal_mode(
        prompt_template,
        client,
        ranker,
        get_problem_tests,
        run_tests,
        num_solutions,
        problem_statement,
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
def build_web_ui(
    _prompt_template,
    _client,
    _ranker,
    _get_problem_tests,
    _run_tests,
    _num_solutions,
    statement=None,
):
    global prompt_template, client, ranker, get_problem_tests, run_tests

    prompt_template = _prompt_template
    client = _client
    ranker = _ranker
    get_problem_tests = _get_problem_tests
    run_tests = _run_tests

    app = gr.Blocks(theme=gr.themes.Soft())

    with app:
        gr.Markdown(
            """
            # φ-Code: A Python Agentic Competitive Programmer Fueled by (tiny) LLMs.
            Click the generate button to get solutions from the model, then use the buttons to browse through them.
            """
        )

        problem_statement = gr.Textbox(
            label="Problem Statement:", value=statement, lines=45
        )
        num_solutions = gr.Number(
            label="Number of Solutions to Generate:", value=_num_solutions, minimum=1
        )

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

    return app

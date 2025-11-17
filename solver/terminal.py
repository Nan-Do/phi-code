from log_config import log
from tqdm import tqdm
from utils import get_completions_from_prompt, DEBUG


def run_terminal_mode(
    prompt_template,
    client,
    ranker,
    get_problem_tests,
    run_tests,
    num_solutions,
    problem_statement,
):
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

    # Log that the process has ended.
    if run_analysis:
        add_s = "s" if len(generated_solutions) != 1 else ""
        log.info(
            f"Process finished, {len(generated_solutions)} solution{add_s} generated"
        )
    else:
        log.info(
            "Process finished, no solutions were generated, please generate new solutions."
        )

    return generated_solutions

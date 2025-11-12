from log_config import log
from openai import OpenAI


DEBUG = False


def get_completions_from_prompt(
    client: OpenAI, prompt: str, num_completions: int = 1
) -> str:
    # Retrieve the code for the given prompt using the given client.
    # We'll keep asking for completions until we reach stop, an error or
    # num_completions are consumed.
    got_stop, continue_completions = False, True
    reasoning_completions, code_completions = [], []
    while continue_completions and (num_completions != 0):
        try:
            # Send the completion request.
            completion = client.chat.completions.create(
                model="coder-model", messages=[{"role": "user", "content": prompt}]
            )
            choice = completion.choices[0]
            response = choice.message
            finish_reason = choice.finish_reason

            # Check the finish reason if the model is asking to continue,
            # to stop, or there has been an error.
            if finish_reason == "stop":
                got_stop = True
                continue_completions = False
            elif finish_reason == "length":
                continue_completions = True
            else:
                log.error(
                    f'Solution generation produced: "{choice.finish_reason}", finising current generation.'
                )
                break

            # Cosume one completion.
            if num_completions > 0:
                num_completions -= 1
            # If it's a reasoning model, store the reasoning tokens.
            if hasattr(response, "reasoning_content"):
                reasoning_completions.append(response.reasoning_content)
            # Store the generated code.
            code_completions.append(response.content)

        except Exception as e:
            log.error(f"An error occurred generating a solution: {e}")
            break

    if DEBUG:
        for reasoning_completion in reasoning_completions:
            print(reasoning_completion)

    # Notify the user if the model was asking for more completions but we reached
    # the limit.
    if continue_completions and (num_completions == 0) and (not got_stop):
        log.info("Solution generation reached the limit of completions without stop.")

    # Strip the markdown in case the model used it in its answer.
    code_body: str = "".join(code_completions)
    if code_body.startswith("```"):
        code_body = "\n".join(code_body.split("\n")[1:-1])
    return code_body

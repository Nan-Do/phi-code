import curses
import sys
import time

from curses import wrapper
from terminal import run_terminal_mode

prompt_template = ""
problem_statement = ""
num_solutions = 0
client, ranker = None, None
get_problem_tests, run_tests = None, None


def show_input_modal(stdscr):
    """Creates a centered modal to prompt the user for an integer."""
    h, w = stdscr.getmaxyx()

    # Define modal dimensions
    MODAL_HEIGHT = 7
    MODAL_WIDTH = 45  # Slightly adjusted width for better centering

    # Calculate starting positions for centering
    start_y = (h - MODAL_HEIGHT) // 2
    start_x = (w - MODAL_WIDTH) // 2

    # Check if there is enough space to draw the modal
    if start_y < 0 or start_x < 0:
        return 0  # Cannot draw modal if screen is too small

    curses.curs_set(1)  # Show cursor for input

    # Create the new window for the modal
    modal_win = curses.newwin(MODAL_HEIGHT, MODAL_WIDTH, start_y, start_x)
    modal_win.box()  # Draw a border
    modal_win.keypad(True)  # Enable special keys like KEY_BACKSPACE

    TEXT_PAIR = curses.color_pair(2)

    # Title
    title = "Change the number of solutions to generate:"
    modal_win.addstr(
        1, (MODAL_WIDTH - len(title)) // 2, title, curses.A_BOLD | TEXT_PAIR
    )

    prompt = "Solutions to generate:"
    modal_win.addstr(3, 2, prompt, TEXT_PAIR)

    user_input = ""
    value = None

    while True:
        # Clear the input line and error line
        modal_win.addstr(4, 2, " " * (MODAL_WIDTH - 4))
        modal_win.addstr(MODAL_HEIGHT - 2, 2, " " * (MODAL_WIDTH - 4))

        # Redraw current input
        modal_win.addstr(4, 2, user_input, TEXT_PAIR)
        modal_win.move(4, 2 + len(user_input))  # Move cursor to end
        modal_win.refresh()

        key = modal_win.getch()

        if key == curses.KEY_ENTER or key in [10, 13]:  # Enter key
            try:
                # Attempt to convert to integer
                value = int(user_input)
                break
            except ValueError:
                # Display error message
                error_msg = "Invalid integer. Press any key to retry."
                modal_win.addstr(
                    MODAL_HEIGHT - 2,
                    2,
                    error_msg,
                    curses.A_REVERSE | curses.color_pair(1),
                )
                modal_win.refresh()
                modal_win.getch()  # Wait for confirmation to dismiss error
                user_input = ""

        elif key == curses.KEY_BACKSPACE or key == 127:  # Backspace
            user_input = user_input[:-1]

        elif 32 <= key < 127:  # Printable ASCII character
            char = chr(key)
            # Only allow to input positive values
            if (char != "0" and char.isdigit()) or (char == "0" and user_input):
                user_input += char

    # Clean up and display result
    curses.curs_set(0)  # Hide cursor

    # Display the confirmed value briefly in the modal area
    confirmation = f"Input confirmed: {value}"

    # Clear the modal area, then print confirmation
    modal_win.clear()
    modal_win.addstr(
        MODAL_HEIGHT // 2,
        (MODAL_WIDTH - len(confirmation)) // 2,
        confirmation,
        curses.A_BOLD | curses.A_REVERSE | TEXT_PAIR,
    )
    modal_win.refresh()

    time.sleep(1)  # Show message for 1 second before closing
    del modal_win

    return value  # Indica


# --- Helper Function to Draw Content in a Curses Window ---
def draw_box_content(win, title, content, title_color_pair, content_color_pair):
    """Draws border, title, and content inside a given curses window."""
    try:
        # Define attributes for content, adding BOLD for the top boxes
        content_attribute = curses.color_pair(content_color_pair) | curses.A_BOLD

        # Draw border
        win.border(0)

        # Print title using the specified color pair and bold attribute
        win.addstr(1, 2, title, curses.A_BOLD | curses.color_pair(title_color_pair))

        # Print content below the title
        # Content is expected to be a single string for the top boxes
        if isinstance(content, str):
            # Apply BOLD attribute to the content of the top boxes
            win.addstr(3, 2, content, content_attribute)
        # Handle list of strings (e.g., code lines) for the main area
        elif isinstance(content, list):
            # For the code box, we don't want the text to be bold,
            # so we only use the color pair here.
            default_attribute = curses.color_pair(content_color_pair)
            for i, line in enumerate(content):
                # We start printing from row 3 (after title and empty line)
                # Ensure we don't write outside the box's inner height
                # win.getmaxyx()[0] is the total height, so we check against Height - 1
                if i + 3 < win.getmaxyx()[0] - 1:
                    if "class" in line or "def" in line:
                        win.addstr(i + 3, 2, line, default_attribute | curses.A_BOLD)
                    else:
                        win.addstr(i + 3, 2, line, default_attribute)

        win.refresh()

    except curses.error as e:
        # Handle errors if the window is too small to display content
        pass


def draw_screen(stdscr, solutions, num_solution, delta_lines):
    # Get screen dimensions
    h, w = stdscr.getmaxyx()

    stdscr.clear()
    stdscr.refresh()

    # Define dimensions for the layout
    TOP_HEIGHT, TOP_WIDTH = 5, w // 3
    CODE_HEIGHT = h - TOP_HEIGHT
    CODE_LOWER_LIMIT = 6

    # --- 2. Top Row Boxes (Status, Tests, Score) ---
    # Box 1: Status (Y=0, X=0)
    win_status = stdscr.subwin(TOP_HEIGHT, TOP_WIDTH, 0, 0)
    status_text = ""
    if solutions:
        status_text = f"Solution {num_solution + 1} of {len(solutions)}"
    draw_box_content(win_status, "🔢 Status:", status_text, 1, 2)

    # Box 2: Tests (Y=0, X=TOP_WIDTH)
    win_tests = stdscr.subwin(TOP_HEIGHT, TOP_WIDTH, 0, TOP_WIDTH)
    tests_text = ""
    if solutions:
        tests_text = f"Passed: {solutions[num_solution]['tests_passed']}"
    draw_box_content(win_tests, "✅ Tests:", tests_text, 1, 2)

    # Box 3: Confidence Score (Y=0, X=2*TOP_WIDTH)
    # Use the remaining width for the third box to ensure it fills the space
    win_score = stdscr.subwin(TOP_HEIGHT, w - (2 * TOP_WIDTH), 0, 2 * TOP_WIDTH)
    score_text = ""
    if solutions:
        score_text = f"{solutions[num_solution]['score']}"
    draw_box_content(win_score, "📈 Confidence Score:", score_text, 1, 2)

    # --- 3. Bottom Code Solution Box ---
    win_code = stdscr.subwin(CODE_HEIGHT, w, TOP_HEIGHT, 0)
    code_view = ""
    if solutions:
        solution_code = solutions[num_solution]["code"]
        start_line, end_line = delta_lines, delta_lines + CODE_HEIGHT - CODE_LOWER_LIMIT
        code_view = solution_code.split("\n")[start_line:end_line]

    # Box 4: Code Solution (Y=TOP_HEIGHT, X=0)
    draw_box_content(win_code, "🖥️ Code Solution:", code_view, 1, 2)
    # Footer
    win_code.addstr(
        CODE_HEIGHT - 2,
        2,
        "Keys: '←','→' Change Solution, '↑','↓' Move Code, 'g' Generate, 'u' Num. Solutions, 's' Save,  'q' exit.",
        curses.A_REVERSE,
    )
    win_code.refresh()


def main(stdscr):
    try:
        """The main curses application function."""
        # --- 1. Initialization and Setup ---
        stdscr.clear()
        stdscr.refresh()

        curses.curs_set(0)

        # Check if terminal supports colors
        if curses.has_colors():
            curses.start_color()
            # Define Color Pairs
            # Pair 1: Blue Title on Black (for contrast)
            curses.init_pair(1, curses.COLOR_BLUE, curses.COLOR_BLACK)
            # Pair 2: White Text on Black (for main content/code)
            curses.init_pair(2, curses.COLOR_WHITE, curses.COLOR_BLACK)

        solutions = run_terminal_mode(
            prompt_template,
            client,
            ranker,
            get_problem_tests,
            run_tests,
            num_solutions,
            problem_statement,
        )

        delta_lines, num_solution, lines_code = (
            0,
            0,
            len(solutions[0]["code"].split("\n")),
        )

        while True:
            draw_screen(stdscr, solutions, num_solution, delta_lines)

            key = stdscr.getch()
            if key in [ord("q"), ord("Q")]:
                break

            elif key in [curses.KEY_UP, ord("k"), ord("K")]:
                if delta_lines > 0:
                    delta_lines -= 1

            elif key in [curses.KEY_DOWN, ord("j"), ord("J")]:
                if delta_lines + 1 < lines_code:
                    delta_lines += 1

            elif key in [curses.KEY_LEFT, ord("h"), ord("H")]:
                if num_solution > 0:
                    num_solution -= 1
                    lines_code = len(solutions[num_solution]["code"].split("\n"))

            elif key in [curses.KEY_RIGHT, ord("l"), ord("L")]:
                if num_solution + 1 < len(solutions):
                    delta_lines = 0
                    num_solution += 1
                    lines_code = len(solutions[num_solution]["code"].split("\n"))
            elif key in [ord("g"), ord("G")]:
                solutions = generate_solutions(stdscr)
            elif key in [ord("u"), ord("U")]:
                global num_solutions
                num_solutions = show_input_modal(stdscr)

    except Exception as e:
        curses.endwin()
        print(f"An error occurred: {e}", file=sys.stderr)


def build_curses_ui(
    _prompt_template,
    _client,
    _ranker,
    _get_problem_tests,
    _run_tests,
    _num_solutions,
    statement,
):
    global \
        prompt_template, \
        client, \
        ranker, \
        get_problem_tests, \
        run_tests, \
        num_solutions, \
        problem_statement

    prompt_template = _prompt_template
    client = _client
    ranker = _ranker
    get_problem_tests = _get_problem_tests
    run_tests = _run_tests
    num_solutions = _num_solutions
    problem_statement = prompt_template.format(statement)

    try:
        wrapper(main)
    except curses.error as e:
        print(
            "Curses initialization error. Is your terminal window too small?",
            file=sys.stderr,
        )
        print(f"Detail: {e}", file=sys.stderr)

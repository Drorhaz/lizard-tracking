import sys

# ANSI colors
COLORS = {
    "red": "\033[91m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "blue": "\033[94m",
    "end": "\033[0m",
}

def print_colored(msg: str, color: str = "green", file=sys.stdout):
    prefix = COLORS.get(color, "")
    suffix = COLORS["end"] if prefix else ""
    print(f"{prefix}{msg}{suffix}", file=file)

def print_info(msg: str): print_colored(msg, "blue")
def print_success(msg: str): print_colored(msg, "green")
def print_warn(msg: str): print_colored(msg, "yellow")
def print_error(msg: str): print_colored(msg, "red", file=sys.stderr)
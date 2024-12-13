import importlib
import os.path
import sys
from datetime import datetime
from pathlib import Path

os.makedirs("outputs", exist_ok=True)


def run_main():
    args = sys.argv[1:]
    file_path = args[0]
    function_arguments = args[1:]

    import_str = ".".join(Path(file_path).with_suffix("").parts)

    print(f"> import {import_str} as module")
    module = importlib.import_module(import_str)

    print(f"> module.main({', '.join(function_arguments)})")
    module.main(*function_arguments)


if __name__ == "__main__":
    run_main()

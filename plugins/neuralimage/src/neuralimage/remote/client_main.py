"""Desktop entrypoint which never preloads local compute libraries."""

import os


def main():
    os.environ["NEURALIMAGE_REMOTE_ONLY"] = "1"
    from neuralimage.main import main as desktop_main

    desktop_main()


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()

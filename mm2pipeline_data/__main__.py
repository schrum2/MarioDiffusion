"""Single entry point for the whole conversion pipeline.

    python -m mm2pipeline_data <command> [options]

Commands (in pipeline order):
    extract         HuggingFace mm2_level dataset -> .bcd files
    toost           .bcd -> level JSON + rendered PNG (via toost.exe)
    json-to-ascii   level JSON -> ASCII grids (+ metadata.json sidecar)
    dataset         build / split the tile-id training dataset
    ascii-to-json   generated ASCII -> level JSON
    swe             level JSON -> playable .swe (SMM: World Engine)

Each command forwards to its stage module, so `python -m mm2pipeline_data extract`
and `python -m mm2pipeline_data.extract` are the same thing. Run any command with
--help for its options.
"""
import sys


def _usage():
    print(__doc__.strip())


def main():
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help", "help"):
        _usage()
        sys.exit(0 if argv else 2)

    command, rest = argv[0], argv[1:]

    # Import lazily so one stage's dependencies (e.g. `datasets` for extract)
    # aren't required just to run a different stage.
    if command == "extract":
        from . import extract
        extract.main(rest)
    elif command == "toost":
        from . import toost
        toost.main(rest)
    elif command == "json-to-ascii":
        from . import ascii
        ascii.main_json_to_ascii(rest)
    elif command == "ascii-to-json":
        from . import ascii
        ascii.main_ascii_to_json(rest)
    elif command == "dataset":
        from . import dataset
        dataset.main(rest)
    elif command == "swe":
        from . import swe
        swe.main(rest)
    else:
        print(f"Unknown command: {command!r}\n")
        _usage()
        sys.exit(2)


if __name__ == "__main__":
    main()

from pathlib import Path
import argparse
from tqdm import tqdm
from mmlv_to_vglc import mmlv_to_grid

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        required=True,
        help="Folder inside MarioDiffusion to save converted VGLC files"
    )
    parser.add_argument(
        "--show_conversions",
        action="store_true",
        help="Print the original per-file status lines (Converted: ...) instead of the default tqdm progress bar"
    )
    args = parser.parse_args()

    def status(msg):
        """Emit a routine per-file status line: shown only with --show_conversions, otherwise the
        progress bar conveys progress. Routed through tqdm.write so it never clobbers an active bar."""
        if args.show_conversions:
            tqdm.write(msg)

    # where downloaded levels already are
    input_dir = Path.home() / "AppData/Local/MegaMaker/Levels"

    # user chooses output folder in repo
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list(input_dir.glob("*.mmlv"))

    print("Found", len(files), "levels")

    success = 0
    failed = 0

    # By default show a tqdm progress bar over the files; --show_conversions disables it and
    # restores the original per-file "Converted:" prints.
    for file in tqdm(files, desc="Converting levels", unit="level", disable=args.show_conversions):
        try:
            lines = mmlv_to_grid(file)

            out_file = output_dir / f"{file.stem}.txt"

            out_file.write_text(
                "\n".join("".join(row) for row in lines) + "\n",
                encoding="utf-8"
            )

            success += 1
            status(f"Converted: {file.name}")

        except Exception as e:
            failed += 1
            # Failures are worth surfacing even in bar mode, so route them through tqdm.write
            # (rather than status()) so they show regardless of --show_conversions.
            tqdm.write(f"FAILED: {file.name} - {e}")

    print("\nDone")
    print("Success:", success)
    print("Failed:", failed)


if __name__ == "__main__":
    main()
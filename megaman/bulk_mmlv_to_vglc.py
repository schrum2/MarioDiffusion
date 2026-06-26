from pathlib import Path
import argparse
from mmlv_to_vglc import mmlv_to_grid

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        required=True,
        help="Folder inside MarioDiffusion to save converted VGLC files"
    )
    args = parser.parse_args()

    # where downloaded levels already are
    input_dir = Path.home() / "AppData/Local/MegaMaker/Levels"

    # user chooses output folder in repo
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list(input_dir.glob("*.mmlv"))

    print("Found", len(files), "levels")

    success = 0
    failed = 0

    for file in files:
        try:
            lines = mmlv_to_grid(file)

            out_file = output_dir / f"{file.stem}.txt"

            out_file.write_text(
                "\n".join("".join(row) for row in lines) + "\n",
                encoding="utf-8"
            )

            success += 1
            print("Converted:", file.name)

        except Exception as e:
            failed += 1
            print("\nFAILED:", file.name)

    print("\nDone")
    print("Success:", success)
    print("Failed:", failed)


if __name__ == "__main__":
    main()
import json
import sys

def combine_json_files(output_file, input_files):
    combined_data = []

    for file in input_files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                if not isinstance(data, list):
                    raise ValueError(f"File {file} does not contain a JSON list.")
                combined_data.extend(data)
        except Exception as e:
            print(f"Error reading {file}: {e}", file=sys.stderr)
            sys.exit(1)

    try:
        with open(output_file, 'w') as f:
            json.dump(combined_data, f, indent=4)
        print(f"Combined data written to {output_file}")
    except Exception as e:
        print(f"Error writing to {output_file}: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python combine_data.py <output_file> <input_file1> <input_file2> ...", file=sys.stderr)
        sys.exit(1)

    output_file = sys.argv[1]
    input_files = sys.argv[2:]

    combine_json_files(output_file, input_files)

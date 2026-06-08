Megaman converters (minimal)
============================

This folder contains small, Python-native utilities to convert between
VGLC-style ASCII Mega Man levels and a simple MMLV JSON representation.

Files:

- `vglc_to_mmlv.py`: Read a VGLC `.txt` file and write a compact `.mmlv.json`.
- `mmlv_to_vglc.py`: Read `.mmlv.json` and reconstruct a lossy ASCII `.txt` file.

Usage examples:

```bash
python -m megaman.vglc_to_mmlv path/to/level.txt            # -> level.txt.mmlv.json
python -m megaman.mmlv_to_vglc path/to/level.txt.mmlv.json  # -> level.txt.reconstructed.txt
```

Example with your file:

```powershell 
python -m megaman.vglc_to_mmlv "file.txt"
```

What this does:

- Reads the ASCII Mega Man level from the VGLC-style text file.
- Converts each character into a numeric tile id and saves the result as JSON.
- Writes a new file named `file.txt.mmlv.json`.

You can then reconstruct a text version from the JSON with:

```powershell
python -m megaman.mmlv_to_vglc "file.txt.mmlv.json"
```

The reverse step is lossy because it recreates the ASCII layout from the saved mapping.

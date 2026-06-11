# Mega Man Maker

Convert Mega Man levels between the [VGLC](https://github.com/TheVGLC/TheVGLC) ASCII format and the native Mega Man Maker `.mmlv` format.

## Scripts

- `vglc_to_mmlv.py`: Read a VGLC `.txt` level and write a `.mmlv` file that Mega Man Maker can open.
- `mmlv_to_vglc.py`: Read a `.mmlv` file and write a simplified VGLC-style `.txt` file.

## Usage

Run all commands from the repo root (`MarioDiffusion`):

```
conda activate myenv
cd C:\Users\<YourName>\Documents\GitHub\MarioDiffusion
```

Convert a VGLC text level to a Mega Man Maker level file:

```
python -m megaman.vglc_to_mmlv myLevel.txt
```

You can also set a level name and author that appear inside Mega Man Maker:

```
python -m megaman.vglc_to_mmlv levels\myLevel.txt --name "My Level" --author "YourName"
```

Convert a `.mmlv` file back to VGLC text:

```
python -m megaman.mmlv_to_vglc levels\SUMegaManStudy001.mmlv
```

Both scripts accept an optional second argument for a custom output path:

```
python -m megaman.vglc_to_mmlv input.txt output.mmlv
python -m megaman.mmlv_to_vglc input.mmlv output.txt
```

## Loading a level in Mega Man Maker

After running `vglc_to_mmlv.py`, copy the output `.mmlv` file to:

```
C:\Users\<YourName>\AppData\Local\MegaMaker\Levels\
```
Then open Mega Man Maker and go to **Level Select** — your level will appear under local levels.

## Automatically loading a level in Mega Man Maker (easier then manually)

you can also automatically load levels from their IDs by running the following two commands:

```
cd C:\Users\your_name\AppData\Local\MegaMaker\Levels
```
then:

```
python -c "import requests,gzip; meta=requests.get('https://api.megamanmaker.com/level/download/ID').json(); open('ID.mmlv','w').write(gzip.decompress(requests.get(meta['location']).content).decode()); print('done')"
```



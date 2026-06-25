# Mega Man Maker

Convert Mega Man levels between the [VGLC](https://github.com/schrum2/TheVGLC) ASCII format and the native Mega Man Maker `.mmlv` format.

## Mega Man Conversions

Both conversions can be done by using these commands in order:

```
cd MM_batch
MegaManMaker.bat
```

Drag and drop a `.mmlv` or `.txt` file when prompted — it will automatically convert to the opposite format.

> **Note:** `python` must be available from your command line before running this script. If you are using a virtual environment, activate it first. The script does not assume any particular Python installation or environment name.

When converting a `.txt` (VGLC format) file to `.mmlv`, the resulting file is automatically copied to your Mega Man Maker levels folder (`%USERPROFILE%\AppData\Local\MegaMaker\Levels`) so it will appear in the "My Levels" section of Mega Man Maker. When converting a `.mmlv` file to `.txt` (VGLC format), the output file is saved alongside the input file and is not copied elsewhere.

## Automatic Level Uploader

Automatic downloads based on ID can be done by running these two commands in order:

```
cd MM_Batch
Auto_Upload_MMMaker.bat
```

Enter a level ID when prompted — it will download directly to your Mega Man Maker levels folder.

## Mega Man Game

You can now use your generated MMLV files to play your desired level using this link: [Mega Man Maker](https://megamanmaker.com/).

Ensure your MMLV file is in the Levels folder — this will ensure your level shows up in the "My Levels" section of Mega Man Maker.

## Bulk level Uploader

This allows you to upload a desired number of levels in bulk (default 100 levels) starting at ID 200,000:

```
cd megaman
Bulk_Download.py
```
Additionally, to upload a desired number of levels use command:

```
cd megaman
Bulk_Download.py --target 
```

Enter a level ID when prompted — it will download directly to your Mega Man Maker levels folder.

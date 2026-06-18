@echo off
set MMM=%LOCALAPPDATA%\MegaMaker\Levels
set /p LEVEL_ID=Downloading level ID: 
python -c "import requests,gzip; meta=requests.get('https://api.megamanmaker.com/level/download/%LEVEL_ID%').json(); open(r'%MMM%\%LEVEL_ID%.mmlv','w').write(gzip.decompress(requests.get(meta['location']).content).decode()); print('done')"
@echo off


cd ..


:: Build a Mega Man dataset carrying captions from several sources at once. Each stage reads the
:: previous JSON and copies scene + metadata + all existing caption keys through, then adds its own,
:: so sources accumulate down the chain (deterministic_captions, then one <model>_captions per LLM).

:: Extract scenes from the VGLC Mega Man levels via path-following
python create_megaman_json_data.py --output datasets\\MM_Levels.json --scan_mode path --limit 800

REM deterministic_captions
python MM_create_ascii_captions.py --dataset datasets\\MM_Levels.json --tileset Game_MM\\MM.json --output datasets\\MM_LevelsAndCaptions-multi.json --caption-mode keyed --caption-key deterministic_captions

REM local inference
python llm_ascii_to_caption.py --levels datasets\\MM_LevelsAndCaptions-multi.json --tileset Game_MM\\MM.json --output datasets\\MM_LevelsAndCaptions-multi.json --llm ollama --model gemma4:26b
python llm_ascii_to_caption.py --levels datasets\\MM_LevelsAndCaptions-multi.json --tileset Game_MM\\MM.json --output datasets\\MM_LevelsAndCaptions-multi.json --llm ollama --model llama3.1:8b

REM cloud inference
:: python llm_ascii_to_caption.py --levels datasets\\MM_LevelsAndCaptions-multi.json --tileset Game_MM\\MM.json --output datasets\\MM_LevelsAndCaptions-multi.json --llm claude --model claude-haiku-4-5
:: python llm_ascii_to_caption.py --levels datasets\\MM_LevelsAndCaptions-multi.json --tileset Game_MM\\MM.json --output datasets\\MM_LevelsAndCaptions-multi.json --llm openai --model gpt-5.1

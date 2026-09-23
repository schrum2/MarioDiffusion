
cd ..

REM python llm_ascii_to_caption.py --levels Game_MMLV\DATA\MMLV_LevelsAndCaptions-regular.json --game MMLV --llm ollama --model qwen3.5:9b --output Game_MMLV\DATA\MMLV_LevelsAndCaptions-llm-qwen.json --num_captions 5

python llm_ascii_to_caption.py --levels Game_MMLV\DATA\MMLV_LevelsAndCaptions-llm-qwen.json --game MMLV --llm ollama --model gemma4:12b --output Game_MMLV\DATA\MMLV_LevelsAndCaptions-llm.json --num_captions 5

python split_data.py --json_file Game_MMLV\DATA\MMLV_LevelsAndCaptions-llm.json --train_pct 0.9 --val_pct 0.05 --test_pct 0.05 --seed 42 --game MMLV

call batch/train-diffusion.bat 0 MMLV llm MMLV CLIP single none 0 300 1 no no gemma4:12b_captions
call batch/train-diffusion.bat 0 MMLV llm MMLV CLIP single none 0 300 3 no no gemma4:12b_captions
call batch/train-diffusion.bat 0 MMLV llm MMLV CLIP single none 0 300 5 no no gemma4:12b_captions


call batch/train-diffusion.bat 0 MMLV llm MMLV CLIP single none 0 300 1 no no qwen3.5:9b_captions
call batch/train-diffusion.bat 0 MMLV llm MMLV CLIP single none 0 300 3 no no qwen3.5:9b_captions
call batch/train-diffusion.bat 0 MMLV llm MMLV CLIP single none 0 300 5 no no qwen3.5:9b_captions















REM call train-diffusion.bat 1 MM2 llm MM2 CLIP single none 0 200 1 no no gemma4:12b_captions
REM call train-diffusion.bat 1 MM2 llm MM2 CLIP single none 0 200 2 no no gemma4:12b_captions
REM call train-diffusion.bat 1 MM2 llm MM2 CLIP single none 0 200 3 no no gemma4:12b_captions
REM call train-diffusion.bat 1 MM2 llm MM2 CLIP single none 0 200 4 no no gemma4:12b_captions
REM call train-diffusion.bat 1 MM2 llm MM2 CLIP single none 0 200 5 no no gemma4:12b_captions



REM call train-diffusion.bat 1 MMLV llm MMLV CLIP single none 0 200 1 no no gemma4:12b_captions
REM call train-diffusion.bat 1 MMLV llm MMLV CLIP single none 0 200 2 no no gemma4:12b_captions
REM call train-diffusion.bat 1 MMLV llm MMLV CLIP single none 0 200 3 no no gemma4:12b_captions
REM call train-diffusion.bat 1 MMLV llm MMLV CLIP single none 0 200 4 no no gemma4:12b_captions
REM call train-diffusion.bat 1 MMLV llm MMLV CLIP single none 0 200 5 no no gemma4:12b_captions
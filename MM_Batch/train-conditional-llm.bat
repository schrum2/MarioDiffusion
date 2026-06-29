REM @echo off
REM Usage: train-conditional-llm.bat <seed> <model> [split]
REM Trains a text-conditional diffusion model on LLM MM captions using a general-purpose pretrained text encoder.
REM <seed>  is optional, defaults to 0
REM <model> should be "MiniLM", "GTE", "CLIP", or "T5", defaults to MiniLM
REM [split] is optional; if "split" is specified, each sentence of a caption gets its own
REM         embedding vector (multiple-sentence mode). Omit it out for single-vector mode.
REM         NOTE: split (multiple-sentence) should not be run with gte.
cd ..

set SEED=%1
if "%SEED%"=="" set SEED=0

REM Pick the pretrained text encoder
set MODEL=%2
if /I "%MODEL%"=="" set MODEL=MiniLM
if /I "%MODEL%"=="MiniLM" set MODEL_NAME=sentence-transformers/multi-qa-MiniLM-L6-cos-v1
if /I "%MODEL%"=="GTE" set MODEL_NAME=Alibaba-NLP/gte-large-en-v1.5
if /I "%MODEL%"=="CLIP" set MODEL_NAME=sentence-transformers/clip-ViT-L-14
if /I "%MODEL%"=="T5" set MODEL_NAME=google/t5-v1_1-base

set SPLIT=%3
if /I "%SPLIT%"=="split" (
    set DIFF_OUTPUT=MM-LLM-conditional-%MODEL%split-%SEED%
    set SPLIT_FLAG=--split_pretrained_sentences
) else (
    set DIFF_OUTPUT=MM-LLM-conditional-%MODEL%-%SEED%
    set SPLIT_FLAG=
)

python train_diffusion.py --text_conditional --game MM-Full --tileset datasets\MM.json --json datasets\MM_LevelsAndLLMCaptions-full-train.json --val_json datasets\MM_LevelsAndLLMCaptions-full-validate.json --multiple_captions --pretrained_language_model "%MODEL_NAME%" --num_epochs 500 --output_dir "%DIFF_OUTPUT%" --seed %SEED% %SPLIT_FLAG%

REM Post-training evaluation 
REM 100 unconditional samples from the trained model
python run_diffusion.py --model_path "%DIFF_OUTPUT%" --num_samples 100 --text_conditional --save_as_json --output_dir "%DIFF_OUTPUT%-unconditional-samples" --game MM-Full

REM Generate a sample for every training caption and save each input prompt next to its generated scene
REM Output: <model>\samples-from-train-captions\all_levels.json
python evaluate_caption_adherence.py --model_path "%DIFF_OUTPUT%" --json datasets\MM_LevelsAndLLMCaptions-full-train.json --num_tiles 41 --no_caption_score --save_as_json --output_dir samples-from-train-captions

REM Same for every test caption
REM Output: <model>\samples-from-test-captions\all_levels.json
python evaluate_caption_adherence.py --model_path "%DIFF_OUTPUT%" --json datasets\MM_LevelsAndLLMCaptions-full-test.json --num_tiles 41 --no_caption_score --save_as_json --output_dir samples-from-test-captions

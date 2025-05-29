@echo off
cd ..

REM Get the first argument as SEED_START, default to 0 if not provided
set SEED_START=%1
if "%SEED_START%"=="" set SEED_START=0

REM Get the second argument as SEED_END, default to SEED_START if not provided
set SEED_END=%2
if "%SEED_END%"=="" set SEED_END=%SEED_START%

REM Loop from SEED_START to SEED_END (inclusive)
for /L %%S in (%SEED_START%,1,%SEED_END%) do (
    set SEED=%%S
    set MLM_OUTPUT=SMB1-MLM-regular%%S
    set DIFF_OUTPUT=SMB1-conditional-regular%%S
    set UNCOND_OUTPUT=SMB1-conditional-regular%%S-unconditional-samples

    call python train_mlm.py --epochs 300 --save_checkpoints --json datasets\SMB1_LevelsAndCaptions-regular-train.json --val_json datasets\SMB1_LevelsAndCaptions-regular-validate.json --test_json datasets\SMB1_LevelsAndCaptions-regular-test.json --pkl datasets\SMB1_Tokenizer-regular.pkl --output_dir %%MLM_OUTPUT%% --seed %%S
    call python train_diffusion.py --augment --text_conditional --output_dir "%%DIFF_OUTPUT%%" --num_epochs 500 --json datasets\SMB1_LevelsAndCaptions-regular-train.json --val_json datasets\SMB1_LevelsAndCaptions-regular-validate.json --pkl datasets\SMB1_Tokenizer-regular.pkl --mlm_model_dir %%MLM_OUTPUT%% --plot_validation_caption_score --seed %%S
    call python run_diffusion.py --model_path %%DIFF_OUTPUT%% --num_samples 100 --text_conditional --save_as_json --output_dir "%%UNCOND_OUTPUT%%"
    call python evaluate_caption_adherence.py --model_path %%DIFF_OUTPUT%% --save_as_json --json datasets\SMB1_LevelsAndCaptions-regular.json --output_dir text-to-level-final
    call python evaluate_caption_adherence.py --model_path %%DIFF_OUTPUT%% --save_as_json --json datasets\SMB1_LevelsAndCaptions-regular.json --compare_checkpoints 
    call python evaluate_caption_adherence.py --model_path %%DIFF_OUTPUT%% --save_as_json --json datasets\SMB1_LevelsAndCaptions-regular-test.json --compare_checkpoints 
    call python evaluate_caption_adherence.py --model_path %%DIFF_OUTPUT%% --save_as_json --json datasets\SMB1_RandomTest-regular.json --compare_checkpoints
)
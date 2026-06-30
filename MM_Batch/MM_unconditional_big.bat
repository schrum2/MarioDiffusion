@echo off

cd ..

python megaman\Bulk_Download.py --target 1000

python megaman\bulk_mmlv_to_vglc.py --output megaman\vglc_out

python create_megaman_json_data.py --scan_mode screen_grid --target_height 32 --target_width 32 --stride_x 16 --stride_y 14 --levels megaman\vglc_out --output megaman\32x32_MMLV_Levels.json

REM this part is just so we populate the "caption" field in each json entry, which is required in train_diffusion even for unconditional models
python MM_create_ascii_captions.py --dataset megaman\32x32_MMLV_Levels.json --tileset datasets\MM.json --output megaman\32x32_MMLV_LevelsAndCaptions.json

python split_data.py --json_file megaman\32x32_MMLV_LevelsAndCaptions.json --game mm-full --train_pct .9 --val_pct .05 --test_pct .05

python train_diffusion.py --mixed_precision bf16 --model_dim 192 --dim_mults 1 2 4 4 --attention_head_dim 8 --down_block_types "DownBlock2D" "AttnDownBlock2D" "AttnDownBlock2D" "AttnDownBlock2D" --up_block_types "AttnUpBlock2D" "AttnUpBlock2D" "AttnUpBlock2D" "UpBlock2D" --augment --batch_size 8 --game MM-Full --json megaman\32x32_MMLV_LevelsAndCaptions-train.json --val_json megaman\32x32_MMLV_LevelsAndCaptions-validate.json --output_dir megaman\MM-unconditional-big --num_epochs 300 --save_image_epochs 10

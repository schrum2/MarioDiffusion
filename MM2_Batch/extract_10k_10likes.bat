cd ..
python -m mm2pipeline_data extract --output_folder MM2_Data\bcd --limit 10000 --skip_3dworld --skip_items --skip_subworld_items --likes 10
python -m mm2pipeline_data toost --input MM2_Data\bcd -o MM2_Data\json --images-output MM2_Data\images
python -m mm2pipeline_data json-to-ascii --input MM2_Data\json --output_folder MM2_Data\ascii
python -m mm2pipeline_data dataset build --input MM2_Data\ascii --output_folder MM2_Data\dataset\dataset.json --tileset Game_MM2\mm2_tileset_we.json --sliding_window --stride 20
python -m mm2pipeline_data dataset split --input MM2_Data\dataset\dataset.json --seed 0


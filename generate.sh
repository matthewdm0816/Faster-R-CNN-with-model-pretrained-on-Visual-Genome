#!/usr/bin/env bash

export OUTPATH="/home/mowentao/vizwiz/dataset_with_label/"
python generate_tsv.py --net res101 --dataset vg --load_dir data/pretrained_model \
    --image_dir "/home/mowentao/vizwiz/dataset/val/*.jpg" \
    --out "${OUTPATH}val_vg.pkl" --cuda \
    --id_txt "/home/mowentao/vizwiz/dataset/val_ids.txt"
python convert_data.py --imgid_list "/home/mowentao/vizwiz/dataset/val_ids.txt"  \
                       --input_file "${OUTPATH}val_vg.pkl"  \
                       --output_file "${OUTPATH}val_vg.npy" \
                       --output_dir "${OUTPATH}val/"
# exit 0 # for debug use
python generate_tsv.py --net res101 --dataset vg --load_dir data/pretrained_model \
    --image_dir "/home/mowentao/vizwiz/dataset/train/*.jpg" \
    --out "${OUTPATH}train_vg.pkl" --cuda \
    --id_txt "/home/mowentao/vizwiz/dataset/train_ids.txt"
python convert_data.py --imgid_list "/home/mowentao/vizwiz/dataset/train_ids.txt"  \
                       --input_file "${OUTPATH}train_vg.pkl"  \
                       --output_file "${OUTPATH}train_vg.npy" \
                       --output_dir "${OUTPATH}train/"
python generate_tsv.py --net res101 --dataset vg --load_dir data/pretrained_model \
    --image_dir "/home/mowentao/vizwiz/dataset/test/*.jpg" \
    --out "${OUTPATH}test_vg.pkl" --cuda \
    --id_txt "/home/mowentao/vizwiz/dataset/test_ids.txt"
python convert_data.py --imgid_list "/home/mowentao/vizwiz/dataset/test_ids.txt"  \
                       --input_file "${OUTPATH}test_vg.pkl"  \
                       --output_dir "${OUTPATH}test/"
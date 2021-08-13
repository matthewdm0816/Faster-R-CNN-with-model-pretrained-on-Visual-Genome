#!/usr/bin/env bash

export OUTPATH="../vqav2/features"
python generate_tsv.py --net res101 --dataset vg --load_dir data/pretrained_model \
    --image_dir "../vqav2/img/val/val2014/*.jpg" \
    --out "${OUTPATH}val_vg.pkl" --cuda \
    --id_txt "../vqav2/img/val_ids.txt"
python convert_data.py --imgid_list "../vqav2/img/val_ids.txt"  \
                       --input_file "${OUTPATH}val_vg.pkl"  \
                       --output_file "${OUTPATH}val_vg.npy" \
                       --output_dir "${OUTPATH}/val/" --concat_bb
# exit 0 # for debug use
# python generate_tsv.py --net res101 --dataset vg --load_dir data/pretrained_model \
#     --image_dir "../vqav2/img/train/train2014/*.jpg" \
#     --out "${OUTPATH}train_vg.pkl" --cuda \
#     --id_txt "../vqav2/img/train_ids.txt"
# python convert_data.py --imgid_list "../vqav2/img/train_ids.txt"  \
#                        --input_file "${OUTPATH}train_vg.pkl"  \
#                        --output_file "${OUTPATH}train_vg.npy" \
#                        --output_dir "${OUTPATH}/train/" --concat_bb
# python generate_tsv.py --net res101 --dataset vg --load_dir data/pretrained_model \
#     --image_dir "../vqav2/img/test/test2015/*.jpg" \
#     --out "${OUTPATH}test_vg.pkl" --cuda \
#     --id_txt "../vqav2/img/test/test2015/test_ids.txt"
# python convert_data.py --imgid_list "../vqav2/img/test/test2015/test_ids.txt"  \
#                        --input_file "${OUTPATH}test_vg.pkl"  \
#                        --output_dir "${OUTPATH}/test/" --concat_bb
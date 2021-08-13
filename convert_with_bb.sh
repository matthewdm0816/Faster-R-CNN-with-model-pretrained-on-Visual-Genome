#!/usr/bin/env bash

export OUTPATH="../vqav2/features"
python convert_data.py --imgid_list "../vqav2/img/val_ids.txt"  \
                       --input_file "${OUTPATH}val_vg.pkl"  \
                       --output_file "${OUTPATH}val_vg.npy" \
                       --output_dir "${OUTPATH}_bb/val/" --concat_bb
python convert_data.py --imgid_list "../vqav2/img/train_ids.txt"  \
                       --input_file "${OUTPATH}train_vg.pkl"  \
                       --output_file "${OUTPATH}train_vg.npy" \
                       --output_dir "${OUTPATH}/train/" --concat_bb
python convert_data.py --imgid_list "../vqav2/img/test/test2015/test_ids.txt"  \
                       --input_file "${OUTPATH}test_vg.pkl"  \
                       --output_dir "${OUTPATH}/test/" --concat_bb
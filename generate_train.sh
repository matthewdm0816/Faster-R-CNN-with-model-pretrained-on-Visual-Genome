python generate_tsv.py --net res101 --dataset vg --load_dir data/pretrained_model \
    --image_dir "/home/mowentao/vizwiz/dataset/train/*.jpg" --out "/home/mowentao/vizwiz/dataset/train_vg.pkl" --cuda \
    --id_txt "/home/mowentao/vizwiz/dataset/train_ids.txt"
python convert_data.py --imgid_list "/home/mowentao/vizwiz/dataset/train_ids.txt"  \
                       --input_file "/home/mowentao/vizwiz/dataset/train_vg.pkl"  \
                       --output_file "/home/mowentao/vizwiz/dataset/train_vg.npy" \
                       --output_dir "/home/mowentao/vizwiz/dataset/features_vg/train/"
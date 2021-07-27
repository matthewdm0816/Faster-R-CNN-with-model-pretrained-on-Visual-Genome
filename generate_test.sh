python generate_tsv.py --net res101 --dataset vg --load_dir data/pretrained_model \
    --image_dir "/home/mowentao/vizwiz/dataset/test/*.jpg" --out "/home/mowentao/vizwiz/dataset/test_vg.pkl" --cuda \
    --id_txt "/home/mowentao/vizwiz/dataset/test_ids.txt"
python convert_data.py --imgid_list "/home/mowentao/vizwiz/dataset/test_ids.txt"  \
                       --input_file "/home/mowentao/vizwiz/dataset/test_vg.pkl"  \
                       --output_dir "/home/mowentao/vizwiz/dataset/features_vg/test/"
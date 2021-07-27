for name in {"VizWiz_train_00011111.jpg","VizWiz_train_00011121.jpg","VizWiz_train_00012111.jpg","VizWiz_train_00011112.jpg"} 
do
python demo.py --net res101 --dataset vg --load_dir data/pretrained_model \
    --cuda --image_dir ../dataset/train --image_file $name  \
    --image_output_folder ~/testimages/out
done
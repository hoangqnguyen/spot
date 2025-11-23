python train_e2e_spatial.py vnl_2.0  data/vnl_2.0/frames_224p  -m rny008_gsm -t gru --clip_len 64 --batch_size 8 --num_epochs 150 -s exp/e2espatial_vnl2.0   --predict_location  --num_workers 4  --wandb_project e2espatial_vnl2.0

# Example: train on the new `hogak` dataset. Replace the `frame_dir` with your
# prepared frames directory (e.g. `data/hogak/volli_dataset_json/frames_1080p` or
# preprocessed `frames_224p` if you have resized frames).
python train_e2e_spatial.py hogak data/hogak/frames_1080p -m rny008_gsm -t gru --clip_len 64 --batch_size 8 --num_epochs 150 -s exp/e2espatial_hogak --predict_location --num_workers 4 --wandb_project e2espatial_hogak

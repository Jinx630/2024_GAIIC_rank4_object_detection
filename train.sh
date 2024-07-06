python code/clean_visdrone_dataset.py --train_annotation_data_dir data/contest_data/ --rgb_image_save_dir 

# cmd-01
python -m torch.distributed.launch --nproc_per_node 4 --master_port 9516 code/train_gelan_gpu2_v3_2mlab.py \
    --workers 8 \
    --batch 64 \
    --data code/data/detect_viscutmorewithoutobj.yaml \
    --img 640 \
    --epochs 1 \
    --cfg code/models/detect/gelan-e-double-img-2mlab.yaml \
    --weights data/pretrain_model/gelan-e.pt \
    --project data/model_data/ckpt-003-2mlab \

# cmd-02
python -m torch.distributed.launch --nproc_per_node 4 --master_port 9516 code/train_gelan_gpu2_v1_4mlab.py \
    --workers 8 \
    --batch 64 \
    --data code/data/detect_viscutmorewithoutobj.yaml \
    --img 640 \
    --epochs 1 \
    --cfg code/models/detect/gelan-e-double-img-mlab.yaml \
    --weights data/pretrain_model/gelan-e.pt \
    --project data/model_data/ckpt-001-4prelatermlab \
    
# cmd-03
python -m torch.distributed.launch --nproc_per_node 4 --master_port 9516 code/train_gelan_gpu2_v2_4mlab_cross.py \
    --workers 8 \
    --batch 64 \
    --data code/data/detect_viscutmorewithoutobj.yaml \
    --img 640 \
    --epochs 1 \
    --cfg code/models/detect/gelan-e-double-img-mlab-cross.yaml \
    --weights data/pretrain_model/gelan-e.pt \
    --project data/model_data/ckpt-002-4prelatercrossmlab \
    
# cmd-04
python -m torch.distributed.launch --nproc_per_node 4 --master_port 9516 code/train_gelan_gpu2_v4_nomlab.py \
    --workers 8 \
    --batch 64 \
    --data code/data/detect_viscutmorewithoutobj.yaml \
    --img 640 \
    --epochs 1 \
    --cfg code/models/detect/gelan-e-double-img.yaml \
    --weights data/pretrain_model/gelan-e.pt \
    --project data/model_data/ckpt-004
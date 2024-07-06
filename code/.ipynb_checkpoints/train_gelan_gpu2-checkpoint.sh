# set -e
# set -x
# project=output
# name=train_gelan_aug

# # 合成save_dir路径
# save_dir="$project/$name"

# # 打印save_dir路径
# echo "save_dir = $save_dir"

# # 检查目录是否存在，如果存在则删除它
# if [ -d "$save_dir" ]; then
#     rm -rf "$save_dir"
# fi
python -m torch.distributed.launch --nproc_per_node 4 --master_port 9516 train_gelan_gpu2_v3_2mlab.py \
    --workers 8 \
    --device 0,1,2,3 \
    --batch 32 \
    --data data/detect_match.yaml \
    --img 960 \
    --cfg models/detect/gelan-e-double-img-2mlab.yaml \
    --weights ../ckpts/yolov9-e.pt \
    --project output/yolov9-e-bs24-match-6cls-randaug-0.3rgbaugaug-0.3tiraugaug-blur-5-2mlab-960size \
    --hyp data/hyps/hyp.scratch-high_copy.yaml \
    --model_size e \
    --save_period 1 \
    --min-items 0 \
    --epochs 100 \
    --custom_aug true \
    --close-mosaic 0
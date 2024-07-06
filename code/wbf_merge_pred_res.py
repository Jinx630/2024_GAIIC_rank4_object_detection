"""
@Time: 2024/5/14 20:58
@Author: xujinlingbj
@File: wbf_merge_pred_res.py
"""
import json
import sys
from collections import defaultdict
from ensemble_boxes import weighted_boxes_fusion
from tqdm import tqdm
import json
from collections import defaultdict


def prepare_and_fuse_predictions(
    pred_file_path_list, iou_thr=0.5, skip_box_thr=0.0001, weights=[1, 1, 1], image_width=640, image_height=512, img_ids=None,
):
    model_pred_res = []
    # Step 1: Aggregate predictions by image_id
    image_id_2_pred_infos = defaultdict(lambda: {"boxes_list": [], "scores_list": [], "labels_list": []})
    all_image_ids = []
    for pred_file_path in pred_file_path_list:
        with open(pred_file_path, "r") as f:
            pred_data = json.load(f)
        # current_pred_bbox = []
        # current_pred_scores = []
        # current_pred_labels = []
        current_image_id2_pred_infos = defaultdict(lambda: {"boxes_list": [], "scores_list": [], "labels_list": []})
        for line in pred_data:
            if line["category_id"] is None:
                continue
            bbox = line["bbox"]
            if bbox[2] == 0.0 or bbox[3] == 0.0:
                continue
            # Assuming bbox format is [x, y, width, height]
            bbox_normalized = [
                bbox[0] / image_width,
                bbox[1] / image_height,
                min((bbox[0] + bbox[2]) / image_width, 1.0),
                min((bbox[1] + bbox[3]) / image_height, 1.0),
            ]  # Convert to [x_min, y_min, x_max, y_max]

            score = line["score"]
            label = line["category_id"]
            current_image_id2_pred_infos[line["image_id"]]["boxes_list"].append(bbox_normalized)
            current_image_id2_pred_infos[line["image_id"]]["scores_list"].append(score)
            current_image_id2_pred_infos[line["image_id"]]["labels_list"].append(label)
        for idd in range(1, 1001):
            pred_info = current_image_id2_pred_infos[idd]
            info = image_id_2_pred_infos[idd]
            info["boxes_list"].append(pred_info["boxes_list"])
            info["scores_list"].append(pred_info["scores_list"])
            info["labels_list"].append(pred_info["labels_list"])

    save_image_ids = set()
    # Step 2: Fuse boxes per image_id
    for image_id, pred_infos in tqdm(image_id_2_pred_infos.items()):
        assert len(pred_infos["boxes_list"]) == len(pred_infos["scores_list"]) and len(
            pred_infos["scores_list"]
        ) == len(weights)
        # print(pred_infos['scores_list'])
        # print(pred_infos['labels_list'])
        boxes, scores, labels = weighted_boxes_fusion(
            pred_infos["boxes_list"],
            pred_infos["scores_list"],
            pred_infos["labels_list"],
            weights=weights,
            iou_thr=iou_thr,
            skip_box_thr=skip_box_thr,
        )
        # Assuming we want to collect results in some format
        for bbox, score, label in zip(boxes, scores, labels):
            bbox[0] *= image_width
            bbox[1] *= image_height
            bbox[2] *= image_width
            bbox[3] *= image_height
            bbox[2] -= bbox[0]
            bbox[3] -= bbox[1]

            model_pred_res.append(
                {
                    "image_id": int(image_id),
                    "bbox": [round(x, 3) for x in bbox],  # Note: Convert back to original format if needed
                    "score": round(score, 4),
                    "category_id": int(label),
                }
            )
            save_image_ids.add(image_id)
    for idd in img_ids:
        if idd not in save_image_ids:
            coco_annotation = {
                "image_id": idd,
                "category_id": None,
                "bbox": None,
                "score": None,
            }
            model_pred_res.append(coco_annotation)

    return model_pred_res


if __name__ == "__main__":
    source_dir = '/mnt/workspace/workgroup/jinmu'
    pred_file_path_list = [
        f"{source_dir}/ai_competition/yolov9_new/output_0608/test_0001/ckpt-001-test/test_epoch50_predictions.json",
        f"{source_dir}/ai_competition/yolov9_new/output_0608/test_0001/ckpt-002-test/test_epoch50_predictions.json",
        f"{source_dir}/ai_competition/yolov9_new/output_0608/test_0001/ckpt-003-test/test_epoch50_predictions.json",
        f"{source_dir}/ai_competition/yolov9_new/output_0608/test_0001/ckpt-004-test/test_epoch50_predictions.json",
        f"{source_dir}/ai_competition/yolov9_new/output_0608/test_0001/ckpt-005-test/test_epoch50_predictions.json",

        f"{source_dir}/ai_competition/yolov9_new/output_0608/test_0001/ckpt-006-test/test_epoch50_predictions.json",
        f"{source_dir}/ai_competition/yolov9_new/output_0608/test_0001/ckpt-007-test/test_epoch50_predictions.json",
        f"{source_dir}/ai_competition/yolov9_new/output_0608/test_0001/ckpt-008-test/test_epoch50_predictions.json",
        f"{source_dir}/ai_competition/yolov9_new/output_0608/test_0001/ckpt-009-test/test_epoch50_predictions.json",
        f"{source_dir}/ai_competition/yolov9_new/output_0608/test_0001/ckpt-010-test/test_epoch50_predictions.json",

        f"{source_dir}/ai_competition/yolov9_new/output_0608/test_0001/ckpt-011-test/test_epoch50_predictions.json",
        f"{source_dir}/ai_competition/yolov9_new/output_0608/test_0001/ckpt-012-test/test_epoch50_predictions.json",
        # f"{source_dir}/ai_competition/yolov9_new/output_0608/test/ckpt-013-test/test_epoch50_predictions.json",
        f"{source_dir}/ai_competition/yolov9_new/output_0608/test_0001/ckpt-014-test/test_epoch50_predictions.json",
        f"{source_dir}/ai_competition/yolov9_new/output_0608/test_0001/ckpt-015-test/test_epoch50_predictions.json",

        f"{source_dir}/ai_competition/yolov9_new/output_0608/test_0001/ckpt-016-test/test_epoch50_predictions.json",
        f"{source_dir}/ai_competition/yolov9_new/output_0608/test_0001/ckpt-017-test/test_epoch50_predictions.json",
        # f"{source_dir}/ai_competition/yolov9_new/output_0608/test/ckpt-018-test/test_epoch50_predictions.json",
        # f"{source_dir}/ai_competition/yolov9_new/output_0608/test/ckpt-019-test/test_epoch50_predictions.json",
        # f"{source_dir}/ai_competition/yolov9_new/output_0608/test/ckpt-020-test/test_epoch50_predictions.json",
        
        f"{source_dir}/ai_competition/yolov9_new/output_0608/test_0001/ckpt-021-test/test_epoch50_predictions.json",
        f"{source_dir}/ai_competition/yolov9_new/output_0608/test_0001/ckpt-022-test/test_epoch50_predictions.json",
        f"{source_dir}/ai_competition/yolov9_new/output_0608/test_0001/ckpt-023-test/test_epoch50_predictions.json",
        f"{source_dir}/ai_competition/yolov9_new/output_0608/test_0001/ckpt-024-test/test_epoch50_predictions.json",
        f"{source_dir}/ai_competition/yolov9_new/output_0608/test_0001/ckpt-025-test/test_epoch50_predictions.json",
    ]
    # 使用示例
    out_path = "ensemble/fused_predictions_new_21_iou6_0001.json"
    iou_thr = 0.6
    skip_box_thr = 0.001
    weights = [2, 2, 2, 2, 2, 
               2, 1, 1, 1, 1, 
               2, 2, 1, 1, 1, 
               1, 2, 2, 1, 1, 
               1]  # 根据实际情况调整权重
    fused_predictions = prepare_and_fuse_predictions(
        pred_file_path_list, iou_thr=iou_thr, skip_box_thr=skip_box_thr, weights=weights
    )

    # 如果需要，可以将融合后的预测结果保存到json文件
    with open(out_path, "w") as f:
        json.dump(fused_predictions, f, ensure_ascii=False)

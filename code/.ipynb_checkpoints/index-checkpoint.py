from json import dump
import os
from val_v9 import run_mlab
from val_ir_rgb import run_layer
from wbf_merge_pred_res import prepare_and_fuse_predictions

def invoke(input_dir, output_path):
    
    test_path_rgb = os.path.join(input_dir, 'rgb')
    files = os.listdir(test_path_rgb)
    files_splied = [file.split('.') for file in files]
    img_ids = [int(split[0]) for split in files_splied if split[-1] == 'jpg']

    batch_size=16
    conf_thres=0.00001
    
    ckpt_path = "../data"
    tmp_path = "../data/tmp_data"
    
    run_mlab(data=input_dir, weights=f'{ckpt_path}/ckpt_001_mlab_test.pt', save_dir=f"{tmp_path}/res_test_001.json", batch_size=batch_size, conf_thres=conf_thres)
    run_mlab(data=input_dir, weights=f'{ckpt_path}/ckpt_002_prelatermlab_test.pt', save_dir=f"{tmp_path}/res_test_002.json", batch_size=batch_size, conf_thres=conf_thres)
    run_mlab(data=input_dir, weights=f'{ckpt_path}/ckpt_003_prelatermlab_test.pt', save_dir=f"{tmp_path}/res_test_003.json", batch_size=batch_size, conf_thres=conf_thres)
    run_mlab(data=input_dir, weights=f'{ckpt_path}/ckpt_004_prelatercrossmlab_test.pt', save_dir=f"{tmp_path}/res_test_004.json", batch_size=batch_size, conf_thres=conf_thres)
    run_mlab(data=input_dir, weights=f'{ckpt_path}/ckpt_005_mlab_test.pt', save_dir=f"{tmp_path}/res_test_005.json", batch_size=batch_size, conf_thres=conf_thres)
    
    run_layer(data=input_dir, weights=f'{ckpt_path}/ckpt_006_test.pt', save_dir=f"{tmp_path}/res_test_006.json", batch_size=batch_size, conf_thres=conf_thres)
    run_layer(data=input_dir, weights=f'{ckpt_path}/ckpt_007_test.pt', save_dir=f"{tmp_path}/res_test_007.json", batch_size=batch_size, conf_thres=conf_thres)

    run_layer(data=input_dir, weights=f'{ckpt_path}/ckpt_009_test.pt', save_dir=f"{tmp_path}/res_test_009.json", batch_size=batch_size, conf_thres=conf_thres)
    run_layer(data=input_dir, weights=f'{ckpt_path}/ckpt_010_test.pt', save_dir=f"{tmp_path}/res_test_010.json", batch_size=batch_size, conf_thres=conf_thres)
    
    
    run_layer(data=input_dir, weights=f'{ckpt_path}/ckpt_013_test.pt', save_dir=f"{tmp_path}/res_test_013.json", batch_size=batch_size, conf_thres=conf_thres)
    run_layer(data=input_dir, weights=f'{ckpt_path}/ckpt_014_test.pt', save_dir=f"{tmp_path}/res_test_014.json", batch_size=batch_size, conf_thres=conf_thres)
    run_layer(data=input_dir, weights=f'{ckpt_path}/ckpt_015_test.pt', save_dir=f"{tmp_path}/res_test_015.json", batch_size=batch_size, conf_thres=conf_thres)
    run_layer(data=input_dir, weights=f'{ckpt_path}/ckpt_016_1024_test.pt', save_dir=f"{tmp_path}/res_test_016.json", batch_size=batch_size, conf_thres=conf_thres, imgsz=1024)
    run_layer(data=input_dir, weights=f'{ckpt_path}/ckpt_017_test.pt', save_dir=f"{tmp_path}/res_test_017.json", batch_size=batch_size, conf_thres=conf_thres)
    
    run_layer(data=input_dir, weights=f'{ckpt_path}/ckpt_023_test.pt', save_dir=f"{tmp_path}/res_test_023.json", batch_size=batch_size, conf_thres=conf_thres)
    run_layer(data=input_dir, weights=f'{ckpt_path}/ckpt_024_test.pt', save_dir=f"{tmp_path}/res_test_024.json", batch_size=batch_size, conf_thres=conf_thres)
    run_layer(data=input_dir, weights=f'{ckpt_path}/ckpt_025_test.pt', save_dir=f"{tmp_path}/res_test_025.json", batch_size=batch_size, conf_thres=conf_thres)
    
    run_mlab(data=input_dir, weights=f'{ckpt_path}/ckpt_030_prelatermlab_960_test.pt', save_dir=f"{tmp_path}/res_test_030.json", batch_size=batch_size, conf_thres=conf_thres, imgsz=960)

    run_mlab(data=input_dir, weights=f'{ckpt_path}/ckpt_032_prelatermlab_960_test.pt', save_dir=f"{tmp_path}/res_test_032.json", batch_size=batch_size, conf_thres=conf_thres, imgsz=960)
    
    run_layer(data=input_dir, weights=f'{ckpt_path}/ckpt_033_test.pt', save_dir=f"{tmp_path}/res_test_033.json", batch_size=batch_size, conf_thres=conf_thres)
    
    run_mlab(data=input_dir, weights=f'{ckpt_path}/ckpt_035_prelatercrossmlab_960_test.pt', save_dir=f"{tmp_path}/res_test_035.json", batch_size=batch_size, conf_thres=conf_thres, imgsz=960)
    run_mlab(data=input_dir, weights=f'{ckpt_path}/ckpt_036_6prelatermlab_960_test.pt', save_dir=f"{tmp_path}/res_test_036.json", batch_size=batch_size, conf_thres=conf_thres, imgsz=960)
    run_mlab(data=input_dir, weights=f'{ckpt_path}/ckpt_037_5_2mlab_960_test.pt', save_dir=f"{tmp_path}/res_test_037.json", batch_size=batch_size, conf_thres=conf_thres, imgsz=960)
    
    run_layer(data=input_dir, weights=f'{ckpt_path}/ckpt_046_test.pt', save_dir=f"{tmp_path}/res_test_046.json", batch_size=batch_size, conf_thres=conf_thres)
    run_layer(data=input_dir, weights=f'{ckpt_path}/ckpt_047_test.pt', save_dir=f"{tmp_path}/res_test_047.json", batch_size=batch_size, conf_thres=conf_thres)
    
    
    
    run_mlab(data=input_dir, weights=f'{ckpt_path}/ckpt_001_mlab_test.pt', save_dir=f"{tmp_path}/res_test_001_v4.json", batch_size=batch_size, conf_thres=conf_thres, dignoal=True)
    run_mlab(data=input_dir, weights=f'{ckpt_path}/ckpt_002_prelatermlab_test.pt', save_dir=f"{tmp_path}/res_test_002_v4.json", batch_size=batch_size, conf_thres=conf_thres, dignoal=True)
    run_mlab(data=input_dir, weights=f'{ckpt_path}/ckpt_003_prelatermlab_test.pt', save_dir=f"{tmp_path}/res_test_003_v4.json", batch_size=batch_size, conf_thres=conf_thres, dignoal=True)
    run_mlab(data=input_dir, weights=f'{ckpt_path}/ckpt_004_prelatercrossmlab_test.pt', save_dir=f"{tmp_path}/res_test_004_v4.json", batch_size=batch_size, conf_thres=conf_thres, dignoal=True)
    run_mlab(data=input_dir, weights=f'{ckpt_path}/ckpt_005_mlab_test.pt', save_dir=f"{tmp_path}/res_test_005_v4.json", batch_size=batch_size, conf_thres=conf_thres, dignoal=True)
    
    run_layer(data=input_dir, weights=f'{ckpt_path}/ckpt_006_test.pt', save_dir=f"{tmp_path}/res_test_006_v4.json", batch_size=batch_size, conf_thres=conf_thres, dignoal=True)
    run_layer(data=input_dir, weights=f'{ckpt_path}/ckpt_007_test.pt', save_dir=f"{tmp_path}/res_test_007_v4.json", batch_size=batch_size, conf_thres=conf_thres, dignoal=True)

    run_layer(data=input_dir, weights=f'{ckpt_path}/ckpt_009_test.pt', save_dir=f"{tmp_path}/res_test_009_v4.json", batch_size=batch_size, conf_thres=conf_thres, dignoal=True)
    run_layer(data=input_dir, weights=f'{ckpt_path}/ckpt_010_test.pt', save_dir=f"{tmp_path}/res_test_010_v4.json", batch_size=batch_size, conf_thres=conf_thres, dignoal=True)
    
    run_layer(data=input_dir, weights=f'{ckpt_path}/ckpt_013_test.pt', save_dir=f"{tmp_path}/res_test_013_v4.json", batch_size=batch_size, conf_thres=conf_thres, dignoal=True)
    run_layer(data=input_dir, weights=f'{ckpt_path}/ckpt_014_test.pt', save_dir=f"{tmp_path}/res_test_014_v4.json", batch_size=batch_size, conf_thres=conf_thres, dignoal=True)
    run_layer(data=input_dir, weights=f'{ckpt_path}/ckpt_015_test.pt', save_dir=f"{tmp_path}/res_test_015_v4.json", batch_size=batch_size, conf_thres=conf_thres, dignoal=True)
    run_layer(data=input_dir, weights=f'{ckpt_path}/ckpt_016_1024_test.pt', save_dir=f"{tmp_path}/res_test_016_v4.json", batch_size=batch_size, conf_thres=conf_thres, imgsz=1024, dignoal=True)
    run_layer(data=input_dir, weights=f'{ckpt_path}/ckpt_017_test.pt', save_dir=f"{tmp_path}/res_test_017_v4.json", batch_size=batch_size, conf_thres=conf_thres, dignoal=True)
    
    run_layer(data=input_dir, weights=f'{ckpt_path}/ckpt_023_test.pt', save_dir=f"{tmp_path}/res_test_023_v4.json", batch_size=batch_size, conf_thres=conf_thres, dignoal=True)
    run_layer(data=input_dir, weights=f'{ckpt_path}/ckpt_024_test.pt', save_dir=f"{tmp_path}/res_test_024_v4.json", batch_size=batch_size, conf_thres=conf_thres, dignoal=True)
    run_layer(data=input_dir, weights=f'{ckpt_path}/ckpt_025_test.pt', save_dir=f"{tmp_path}/res_test_025_v4.json", batch_size=batch_size, conf_thres=conf_thres, dignoal=True)
    
    run_mlab(data=input_dir, weights=f'{ckpt_path}/ckpt_030_prelatermlab_960_test.pt', save_dir=f"{tmp_path}/res_test_030_v4.json", batch_size=batch_size, conf_thres=conf_thres, imgsz=960, dignoal=True)

    run_mlab(data=input_dir, weights=f'{ckpt_path}/ckpt_032_prelatermlab_960_test.pt', save_dir=f"{tmp_path}/res_test_032_v4.json", batch_size=batch_size, conf_thres=conf_thres, imgsz=960, dignoal=True)
    
    run_layer(data=input_dir, weights=f'{ckpt_path}/ckpt_033_test.pt', save_dir=f"{tmp_path}/res_test_033_v4.json", batch_size=batch_size, conf_thres=conf_thres, dignoal=True)
    
    run_mlab(data=input_dir, weights=f'{ckpt_path}/ckpt_035_prelatercrossmlab_960_test.pt', save_dir=f"{tmp_path}/res_test_035_v4.json", batch_size=batch_size, conf_thres=conf_thres, dignoal=True, imgsz=960)
    run_mlab(data=input_dir, weights=f'{ckpt_path}/ckpt_036_6prelatermlab_960_test.pt', save_dir=f"{tmp_path}/res_test_036_v4.json", batch_size=batch_size, conf_thres=conf_thres, dignoal=True, imgsz=960)
    run_mlab(data=input_dir, weights=f'{ckpt_path}/ckpt_037_5_2mlab_960_test.pt', save_dir=f"{tmp_path}/res_test_037_v4.json", batch_size=batch_size, conf_thres=conf_thres, dignoal=True, imgsz=960)
    
    run_layer(data=input_dir, weights=f'{ckpt_path}/ckpt_046_test.pt', save_dir=f"{tmp_path}/res_test_046_v4.json", batch_size=batch_size, conf_thres=conf_thres, dignoal=True)
    run_layer(data=input_dir, weights=f'{ckpt_path}/ckpt_047_test.pt', save_dir=f"{tmp_path}/res_test_047_v4.json", batch_size=batch_size, conf_thres=conf_thres, dignoal=True)
    
    print("=============================== Ensembel ===============================")
    
    pred_file_path_list = [
        f"{tmp_path}/res_test_001.json",
        f"{tmp_path}/res_test_002.json",
        f"{tmp_path}/res_test_003.json",
        f"{tmp_path}/res_test_004.json",
        f"{tmp_path}/res_test_005.json",
        
        f"{tmp_path}/res_test_006.json",
        f"{tmp_path}/res_test_007.json",
        # f"{tmp_path}/res_test_008.json",
        f"{tmp_path}/res_test_009.json",
        f"{tmp_path}/res_test_010.json",
        
        # f"{tmp_path}/res_test_011.json",
        # f"{tmp_path}/res_test_012.json",
        f"{tmp_path}/res_test_013.json",
        f"{tmp_path}/res_test_014.json",
        f"{tmp_path}/res_test_015.json",
        
        f"{tmp_path}/res_test_016.json",
        f"{tmp_path}/res_test_017.json",
        # f"{tmp_path}/res_test_021.json",
        # f"{tmp_path}/res_test_022.json",
        f"{tmp_path}/res_test_023.json",
        
        f"{tmp_path}/res_test_024.json",
        f"{tmp_path}/res_test_025.json",
        # f"{tmp_path}/res_test_026.json",
        # f"{tmp_path}/res_test_027.json",
        f"{tmp_path}/res_test_030.json",
        
        # f"{tmp_path}/res_test_031.json",
        f"{tmp_path}/res_test_032.json",
        f"{tmp_path}/res_test_033.json",
        # f"{tmp_path}/res_test_034.json",
        
        f"{tmp_path}/res_test_035.json",
        f"{tmp_path}/res_test_036.json",
        f"{tmp_path}/res_test_037.json",
        
        # f"{tmp_path}/res_test_040.json",
        # f"{tmp_path}/res_test_042.json",
        # f"{tmp_path}/res_test_044.json",
        
        f"{tmp_path}/res_test_046.json",
        f"{tmp_path}/res_test_047.json",
        
        
        f"{tmp_path}/res_test_001_v4.json",
        f"{tmp_path}/res_test_002_v4.json",
        f"{tmp_path}/res_test_003_v4.json",
        f"{tmp_path}/res_test_004_v4.json",
        f"{tmp_path}/res_test_005_v4.json",
        
        f"{tmp_path}/res_test_006_v4.json",
        f"{tmp_path}/res_test_007_v4.json",
        # f"{tmp_path}/res_test_008.json",
        f"{tmp_path}/res_test_009_v4.json",
        f"{tmp_path}/res_test_010_v4.json",
        
        # f"{tmp_path}/res_test_011.json",
        # f"{tmp_path}/res_test_012.json",
        f"{tmp_path}/res_test_013_v4.json",
        f"{tmp_path}/res_test_014_v4.json",
        f"{tmp_path}/res_test_015_v4.json",
        
        f"{tmp_path}/res_test_016_v4.json",
        f"{tmp_path}/res_test_017_v4.json",
        # f"{tmp_path}/res_test_021.json",
        # f"{tmp_path}/res_test_022.json",
        f"{tmp_path}/res_test_023_v4.json",
        
        f"{tmp_path}/res_test_024_v4.json",
        f"{tmp_path}/res_test_025_v4.json",
        # f"{tmp_path}/res_test_026.json",
        # f"{tmp_path}/res_test_027.json",
        f"{tmp_path}/res_test_030_v4.json",
        
        # f"{tmp_path}/res_test_031.json",
        f"{tmp_path}/res_test_032_v4.json",
        f"{tmp_path}/res_test_033_v4.json",
        # f"{tmp_path}/res_test_034.json",
        
        f"{tmp_path}/res_test_035_v4.json",
        f"{tmp_path}/res_test_036_v4.json",
        f"{tmp_path}/res_test_037_v4.json",
        
        # f"{tmp_path}/res_test_040_v4.json",
        # f"{tmp_path}/res_test_042_v4.json",
        # f"{tmp_path}/res_test_044_v4.json",
        
        f"{tmp_path}/res_test_046_v4.json",
        f"{tmp_path}/res_test_047_v4.json",
    ]
    # 使用示例
    iou_thr = 0.8
    skip_box_thr = 0.001
    weights = [2, 2, 2, 2, 2, 
               2, 1, 1, 1, 1, 
               1, 1, 1, 1, 1, 
               1, 1, 2, 2, 2,
               2, 2, 2, 1, 1,
               
               2, 2, 2, 2, 2, 
               2, 1, 1, 1, 1, 
               1, 1, 1, 1, 1, 
               1, 1, 2, 2, 2,
               2, 2, 2, 1, 1]  # 根据实际情况调整权重
    
    fused_predictions = prepare_and_fuse_predictions(
        pred_file_path_list, iou_thr=iou_thr, skip_box_thr=skip_box_thr, weights=weights, img_ids=img_ids
    )

    # 如果需要，可以将融合后的预测结果保存到json文件
    with open(output_path, "w") as f:
        dump(fused_predictions, f, ensure_ascii=False)

if __name__ == '__main__':
    data_path = "../data/contest_data/test"
    output_path = "../data/result/result.json"
    invoke(data_path, output_path)
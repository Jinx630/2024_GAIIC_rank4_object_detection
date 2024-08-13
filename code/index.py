from json import dump
import os
from val import run_mlab

def invoke(input_dir, output_path, task, weight):
    
    test_path_rgb = os.path.join(input_dir, 'rgb')
    files = os.listdir(test_path_rgb)
    files_splied = [file.split('.') for file in files]
    img_ids = [int(split[0]) for split in files_splied if split[-1] == 'jpg']

    batch_size=16
    conf_thres=0.0001
    
    run_mlab(
        data=input_dir,
        weights=weight,
        save_dir=output_path,
        batch_size=batch_size,
        conf_thres=conf_thres,
        task=task
    )
    
    print("=============================== save pred. ===============================")
    
if __name__ == '__main__':
    task = 'val'
    data_path = f"data/contest_data/{task}"
    output_path = f"data/result/result_{task}.json"
    weight = 'ckpt_002_prelatermlab_test.pt'
    invoke(data_path, output_path, task, weight)
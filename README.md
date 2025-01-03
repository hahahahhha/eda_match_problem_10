# 目录结构

|   .gitignore
|   analyze.py
|   dataset_convert.py
|   pic_resize.py
|   README.md
|   read_annotations.py
|   report.md
|   yolo11n.pt
|   yolo11_train.py
|   yolo_train.yaml
|   
+---6th_integrated_circuit_eda_elite_challenge_question10_dataset
|           
+---datasets
|   |               
|   \---eda
|       |   eda.yaml
|       |   
|       +---images
|       |   +---test
|       |   +---train
|       |   \---val
|       |           
|       +---labels
|       |   |   train.cache
|       |   |   val.cache
|       |   |   
|       |   +---test
|       |   |       
|       |   +---train
|       |   |       
|       |   \---val
|       |           
|       +---yolo_train_run
|       +---yolo_train_run2
|       \---yolo_train_run3
|                   
\---public
    |   GED4py-master.zip
    |   main.py
    |   README-1026更新.pdf
    |   README.md
    |   utils.py
    |   图1.png
    |   图2.png
    |   图3.png
    |   validation_report.md
    |
    +---GED4py-master
    +---generate
    +---images
    +---true
    |       
    \---__pycache__

# 使用说明 
如果想重新训练 yolo11 模型，需要手动修改 datasets/eda/eda.yaml 中的 path 为当前的 datasets/eda 的绝对目录

# 实现方法
            

# 多模态情感分析
图像+文本数据的双模态情感分析分类问题

## Setup

This implemetation is based on Python3. To run the code, you need the following dependencies:

- torch==1.10.0

- sklearn==1.0.0

- transformers==4.30.2

使用指令



## Repository structure
```python
|-- data/ # 图片数据和文本数据（在此处略去）
|-- img/ #图片储存地
|-- predict/ # （样例）先前预测的结果，包括消融实验
|-- data.txt # 训练模型时测试和训练准确率的储存地
|-- img.ipynb # 折线图片生成地
|-- main.py # 主要流程代码
|-- model.py # OTEmodel.py
|-- README.md
|-- requirements.txt #基本环境 
|-- sampled.txt # data.txt的范例
|-- setup.py #main.py主要流程中需要用到的一些函数以及数据结构
|-- test_without_label.txt  #测试集
|-- train.txt # 训练集


2,1,0
分别对应mix，text，image
运行程序就是  python main.py (-o 2|1|0)|null



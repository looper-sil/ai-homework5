import pandas as pd
import cv2
import numpy as np
import torch
import torch.nn as nn
import argparse
#import matplotlib.pyplot as plt


from transformers import BertModel, BertTokenizer
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50

from model import *
from setup import *

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
pretrained_model = BertModel.from_pretrained("bert-base-multilingual-cased")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = "./data/"

text_length_most = 131 
uid=3
lr = [0.00001,0.00002,0.000015]
batch_size = 32
best_acc = 0
epochs = 7

#epoch_list=[]
#for i in range(epochs):
#    epoch_list.append(i+1)
#print(epoch_list)    


transform = transforms.Compose([
    transforms.Resize((256, 256)),  
    transforms.ToTensor(),  
])

# 模型训练和验证
torch.cuda.set_device(0)
criterion = nn.CrossEntropyLoss()

#选择模型
option = choose_model()
if option ==0:
    print("only use image train")
if option ==1:
    print("only use text train")
if option ==2:
    print("use mix model train")

# 数据准备
train_label_path = "./train.txt"
train_label_df = pd.read_csv(train_label_path,sep=",")

column_dict = {
    "positive": 0, 
    "negative": 1,
    "neutral":2
    }

#翻转
column_dict_anti = {
     0:"positive", 
     1:"negative",
     2:"neutral"
     }

new_df = train_label_df.replace({"tag": column_dict})
labels = list(new_df['tag'])

image_paths = get_paths_according_to_train(path,new_df)
texts = get_texts_according_to_train(path,new_df)

#训练测试分开，分别得到路径，词元，标签
image_paths_train, image_paths_val, texts_train, texts_val, labels_train, labels_val = train_test_split(
    image_paths, texts, labels, test_size=0.2, random_state=5)

#处理得到词元
tokenized_texts_train = text_preprocess(texts_train)
tokenized_texts_val = text_preprocess(texts_val)

#加载得到数据表
dataset_train = Dataset(image_paths_train, tokenized_texts_train, labels_train, transform)
dataset_val = Dataset(image_paths_val,tokenized_texts_val, labels_val, transform)

#构建数据加载器
loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

#读入当前的option
with open('data.txt','a+') as f:    
    f.write('option:'+str(option)+'\n')
    f.close

for lr in lr:
    model = Model(uid,option)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #读入当前的学习率
    with open('data.txt','a+') as f:    
        f.write('lr:'+str(lr)+'\n'+'\t')
        f.close
    train_acc_list=[]
    test_acc_list=[]
    for epoch in range(epochs):

        train_loss, train_acc = train_model(model, loader_train, criterion, optimizer, device)
        val_predictions = predict_model(model, loader_val, device)
        # 计算验证集准确率    
        val_predictions = np.array(val_predictions)
        val_labels = np.array(labels_val)
        val_acc = (val_predictions == val_labels).sum() / len(val_labels)
        
        train_acc_list.append(train_acc)
        test_acc_list.append(val_acc)
        
        if(val_acc>best_acc):
            best_acc = val_acc
            #保存当前在验证集上表现最好的模型
            if(option==0):
                torch.save(model, 'best_image_model.pt')
            if(option==1):
                torch.save(model, 'best_text_mdoel.pt')
            if(option==2):
                torch.save(model, "best_mix_model.pt")
        print(f"batch size: {batch_size}, lr: {lr}, Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, test Acc: {val_acc:.4f}, Best test Acc:{best_acc:.4f}")
    #对于每个学习率的具体数据
    with open('data.txt','a+') as f:    
                f.write('train_acc: ')
                f.close
    for data in train_acc_list:
            with open('data.txt','a+') as f:    
                f.write(format(float(data),'.4f')+',')
                f.close
    with open('data.txt','a+') as f:    
                f.write('\n'+'\t')
                f.close
    with open('data.txt','a+') as f:    
                f.write('test_acc: ')
                f.close
    for data in test_acc_list:
             with open('data.txt','a+') as f:    
                f.write(format(float(data),'.4f')+',')
                f.close
    with open('data.txt','a+') as f:    
                f.write('\n')
                f.close
with open('data.txt','a+') as f:    
            f.write('\n')
            f.close

print("training is over")
print("start predicting")

test_path = "test_without_label.txt"
test_df = pd.read_csv(test_path,sep=",")
test_df['tag']=0
test_labels = np.array(test_df['tag'])

image_paths_test = get_paths_according_to_train(path,test_df)
test_texts = get_texts_according_to_train(path,test_df)
tokenized_texts_test = text_preprocess(test_texts)
dataset_test = Dataset(image_paths_test, tokenized_texts_test, test_labels, transform)
loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

if(option==0):
        best_model = torch.load('best_image_model.pt').to(device)
if(option==1):
        best_model = torch.load('best_text_model.pt').to(device)
if(option==2):
        best_model = torch.load('best_mix_model.pt').to(device)

test_predictions = predict_model(best_model, loader_test, device)  
test_predictions = np.array(test_predictions)

test_df['tag'] = test_predictions
pre_df = test_df.replace({"tag": column_dict_anti})
pre_df.to_csv('predict.txt',sep=',',index=False)
#pre_df.to_csv('test_without_label.txt',sep=',',index=False)
print("prediction finished")
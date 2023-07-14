import pandas as pd
import cv2
import numpy as np
import torch
import torch.nn as nn
import argparse


from transformers import BertModel, BertTokenizer
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50
from model import Imagemodel,Textmodel,Model


tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
text_length_most = 131 


def choose_model():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', type=int, default=2, help='0-only image 1-only text 2-mix') 
    args = parser.parse_args()
    return args.o


def get_paths_according_to_train(path ,df):
    image_paths = []
    for uid in df['guid']:
        image_path = path+str(uid)+".jpg"
        try:
            image = cv2.imread(image_path)
            image_paths.append(image_path)
            height,width,channels = image.shape
        except Exception :
            continue
    return image_paths


def get_texts_according_to_train(path,df):
    texts=[]
    for uid in df['guid']:
        file = path+str(uid)+".txt"
        try:
            with open(file, "r",encoding="GB18030") as f:
                content = f.read()
                texts.append(content)
        except FileNotFoundError:
            continue
    return texts


def text_preprocess(texts):
    
    #清洗数据
    for i in range(len(texts)):
        words = texts[i].split() 
        texts[i]=''
        for word in words:
            if word.startswith('@') or  word.startswith('#') or word.startswith('http') or word.startswith('|'):
                continue
            texts[i]+=word+' ' 
        
    tokenized_texts = [tokenizer(text,padding='text_length_most',text_length_most=text_length_most,truncation=True,return_tensors="pt") for text in texts]
    return tokenized_texts

#图片和文本混合数据集
class Dataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, tokenized_texts, labels,transform=None):
        self.image_paths = image_paths     
        self.transform = transform
        self.input_ids = [x['input_ids'] for x in tokenized_texts]
        self.attention_mask = [x['attention_mask'] for x in tokenized_texts]
        self.labels = labels

    def __getitem__(self, index):
        input_ids = torch.tensor(self.input_ids[index])
        attention_mask = torch.tensor(self.attention_mask[index])
        labels = torch.tensor(self.labels[index])

        image_path = self.image_paths[index]
        image = Image.open(image_path)
        image = self.transform(image)
        
        return image ,input_ids, attention_mask, labels
    def __len__(self):
        return len(self.input_ids)
       
# 训练过程
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()  
    running_loss = 0
    total_correct = 0 
    for images, input_ids, attention_mask, labels in train_loader:
        images = images.to(device)
        input_ids = input_ids.squeeze(1).to(device)
        attention_mask = attention_mask.to(device)     
        labels = labels.to(device)     
        optimizer.zero_grad()     
        outputs = model(images, input_ids,attention_mask)
        _, preds = torch.max(outputs, 1)
        total_correct += torch.sum(preds == labels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()   
        running_loss += loss.item()
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = total_correct.item() / len(train_loader.dataset)
    return epoch_loss, epoch_acc

# 预测过程
def predict_model(model, test_loader, device):
    model.eval()
    predictions = []
    for images,input_ids, attention_mask,  _ in test_loader:
        images = images.to(device)
        input_ids = input_ids.squeeze(1).to(device)
        attention_mask = attention_mask.to(device)
        with torch.no_grad():
            outputs = model(images, input_ids,attention_mask)
            _, preds = torch.max(outputs, 1)
        predictions.extend(preds.cpu().numpy())
    return predictions
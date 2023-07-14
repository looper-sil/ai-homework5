import torch.nn as nn
import torch

from torchvision.models import resnet50
from transformers import BertModel, BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
pretrained_model = BertModel.from_pretrained("bert-base-multilingual-cased")

class Imagemodel(nn.Module):
    def __init__(self):
        super(Imagemodel, self).__init__()
        self.resnet = resnet50(pretrained=True)  
    
    def forward(self, image):
        features = self.resnet(image)
        return features
    
class Textmodel(nn.Module):
    def __init__(self):
        super(Textmodel, self).__init__()
        self.bert = pretrained_model

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  
        output = pooled_output
        return output
    

class Model(nn.Module):
    def __init__(self, uid,option):
        super(Model, self).__init__()
        self.Imagemodel = Imagemodel()  
        self.text_encoder = Textmodel()
        self.option=option
        #仅输入图像特征
        self.classifier0 = nn.Sequential(
            nn.Dropout(p=0.6),
            nn.Linear(1000, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.6),
            nn.Linear(256, uid),
            nn.ReLU(inplace=True),
           
        )
        #仅输入文本特征
        self.classifier1 = nn.Sequential(
            nn.Dropout(p=0.6),
            nn.Linear(768, 256),
            nn.ReLU(inplace=True),
            
            nn.Dropout(p=0.6),
            nn.Linear(256, uid),
            nn.ReLU(inplace=True),
        )
        #多模态融合
        self.classifier2 = nn.Sequential(
            nn.Dropout(p=0.6),
            nn.Linear(1768, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.6),
            nn.Linear(1024, uid),
            nn.ReLU(inplace=True),
        )

    
    def forward(self, image, input_ids,attention_mask):
        if(self.option==0):
            image_features = self.Imagemodel(image)

            output = image_features
            output = self.classifier0(image_features)
        if(self.option==1):
            text_features = self.text_encoder(input_ids, attention_mask)

            output = self.classifier1(text_features)
        if(self.option==2):
            image_features = self.Imagemodel(image)
            text_features = self.text_encoder(input_ids,attention_mask)
            mix_features = torch.cat((text_features,image_features), dim=-1)
            
            output = self.classifier2(mix_features)
        return output
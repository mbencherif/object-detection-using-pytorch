import torch
import pandas as pd
from PIL import Image


class PascalDataset(torch.utils.data.Dataset):
    def __init__(self, csv_dir, img_dir, transforms):
        super(PascalDataset, self).__init__()
        self.csv_dir = csv_dir
        self.img_dir = img_dir
        self.transforms = transforms
        
        self.df = pd.read_csv(csv_dir)
        self.len = len(self.df.index)
        categories = self.df['category'].unique()
        
        self.cat_to_id = {}
        self.id_to_cat = {}
        #to generate category ids from 0 -> len(categories)-1
        #need this while calculating loss on preds in training loop
        for i, cat in enumerate(categories):
            self.cat_to_id[cat] = i         
            self.id_to_cat[i] = cat
           
    def __getitem__(self, index):
        img = Image.open(self.img_dir/self.df.loc[index]['file_name'])
        img = self.transforms(img)
        
        label = self.cat_to_id[self.df.loc[index]['category']]
        return (img, label)
    
    def __len__(self):
        return self.len    
    
    def get_category_label(self, id):
        return self.id_to_cat[id]

    
class PascalBboxDataset(torch.utils.data.Dataset):
    def __init__(self, csv_dir, img_dir, transforms):
        super(PascalBboxDataset, self).__init__()
        self.csv_dir = csv_dir
        self.img_dir = img_dir
        
        
        self.df = pd.read_csv(csv_dir)
        self.len = len(self.df.index)
        self.tfms = transforms
           
    def __getitem__(self, index):
        img = Image.open(self.img_dir/self.df.loc[index]['fn'])
        bbox_string = self.df.loc[index]['bbox']
        bbox = [int(val) for val in bbox_string.split(' ')]
        
        item = self.tfms({ 'image' : img, 'bbox' : bbox })
        item['bbox'] = item['bbox'].type(torch.FloatTensor)
        return (item['image'], item['bbox'])
    
    def __len__(self):
        return self.len    
    
    def get_category_label(self, id):
        return self.id_to_cat[id]
    
class ConcatDS(torch.utils.data.Dataset):
    def __init__(self, bbox_ds, classifier_ds):
        super(ConcatDS, self).__init__()
        self.bbox_ds = bbox_ds
        self.classifier_ds = classifier_ds
        
        self.get_category_label = self.classifier_ds.get_category_label
        self.total_categories = len(self.classifier_ds.id_to_cat.items())
        
        self.len = len(classifier_ds)
           
    def __getitem__(self, index):
        _, cat_id = self.classifier_ds[index]
        img, bbox = self.bbox_ds[index]
        
        return (img, (bbox, cat_id))
    
    def __len__(self):
        return self.len        
    
class PascalMultiClassDataset(PascalDataset):
    def __init__(self, csv_dir, img_dir, category_dic, transforms):
        super(PascalMultiClassDataset, self).__init__(csv_dir, img_dir, transforms)
        
        self.cat_to_id = {}
        self.id_to_cat = {}
        self.categories_len = len(category_dic.items())
        
        for i, id in enumerate(category_dic):
            self.cat_to_id[category_dic[id]] = i
            self.id_to_cat[i] = category_dic[id]   
        
    def __getitem__(self, index):
        img = Image.open(self.img_dir/self.df.loc[index]['file_name'])
        img = self.transforms(img)
        
        cat_ids = [ self.cat_to_id[cat] for cat in self.df.loc[index]['category'].split(' ') ]
        one_hot_encoded = torch.zeros((self.categories_len))
        one_hot_encoded[cat_ids] = 1
        return (img, one_hot_encoded)
    
    def get_cat_labels(self, logits):
        cats = []
        for i, val in enumerate(logits):
            if(bool(val.item())):
                cats.append(self.id_to_cat[i])
        return ' '.join(cats)

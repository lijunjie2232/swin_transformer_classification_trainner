from torch.utils.data import Dataset
import pandas as pd
import os
from tqdm import tqdm
import re
from utils import readImage
import numpy as np

class MyDataset(Dataset):
    def __init__(self, datasetPath, dataType='', fileNameForm=None, imageProcessor=None, updateIndex=False, tag2label:map=None, tags:list=None, saveConfig=False):
        """
        dataType: ['train', 'val', 'test']
        """
        if dataType:
            assert dataType in ['train', 'val', 'test']
            self.imageDataPath = os.path.join(datasetPath, dataType, 'images')
        else:
            self.imageDataPath = os.path.join(datasetPath, 'images')
        assert os.path.isdir(self.imageDataPath)
        
        self.imageProcessor = imageProcessor
        self.dataType = dataType
        
        self.tags = ['tent', 'car', 'truck', 'human', 'bridge', 'bg']
        self.sufList = ['.jpg', '.png']
        if tags:
            self.tags = tags
        if tag2label:
            self.tag2label = tag2label
        else:
            self.tag2label = {self.tags[i]:i for i in range(len(self.tags))}
            self.label2tag = {i:self.tags[i] for i in range(len(self.tags))}
        # self.__len__ = self.__len__()
        
        self.fileNameForm = fileNameForm
        if updateIndex:
            self.dataIndex = self.makeIndexFile(self.imageDataPath)  # update data index
        else:
            # assert os.path.isfile(os.path.join(self.imageDataPath, 'index.csv'))
            # self.dataIndex = pd.read_csv(os.path.join(self.imageDataPath, 'index.csv')) # read data index file
            if  os.path.isfile(os.path.join(self.imageDataPath, 'metadata.csv')):
                self.dataIndex = pd.read_csv(os.path.join(self.imageDataPath, 'metadata.csv')) # read data index file
            else:
                self.dataIndex = self.makeIndexFile(self.imageDataPath)  # update data index

            
            
    def __getitem__(self, index):
        # fileName, tag = 
        row = self.dataIndex.iloc[index]
        fileName, imageFilePath, label = row
        # img = cv2.imread(item.split(' _')[0])
        # imageFilePath = os.path.join(self.imageDataPath, fileName)
        img = None
        try:
            img = readImage(imageFilePath)
            if self.imageProcessor:
                img = self.imageProcessor(img, return_tensors="pt")
                img = img['pixel_values']
        except Exception as e:
            print(e)
            return self.__getitem__(np.random.randint(0, self.dataIndex.shape[0], size=None, dtype='l'))
        tag = self.label2tag[label]
        # return {
        #    # 'image': img,
        #    'image': img,
        #    'image_file_path': imageFilePath,
        #    'labels': label,
        #    # 'tag': tag
        # }
        return img, label

    def __len__(self):
        return self.dataIndex.shape[0]

#     def makeIndexFile(self, path:str):
#         data = []
#         fineNamePattern = re.compile(self.fileNameForm)#r'([\w]{32})-(\w.*)\.[\w.*?]'
        
#         os.system('rm -rf %s'%os.path.join(path,'.ipynb_checkpoints'))

#         for fileName in tqdm(os.listdir(path)):
#             result = re.findall(fineNamePattern, fileName)
#             if result:
#                 # data.append((fileName, result[0][0],  result[0][1]))
#                 data.append((fileName, result[0][1]))
        
#         df = pd.DataFrame(data=data, columns = ['fileName', 'tag'])
#         df.to_csv(os.path.join(path, 'index.csv'), index=False)
#         return df

    # def makeMetadataFile(self, path:str):
    def makeIndexFile(self, path:str):
        data = []
        # fineNamePattern = re.compile(self.fileNameForm)
        
        # os.system('rm -rf %s'%os.path.join(path,'.ipynb_checkpoints'))

        # for fileName in tqdm(os.listdir(path)):
            # result = re.findall(fineNamePattern, fileName)
            # if result:
            #     # data.append((fileName, result[0][0],  result[0][1]))
            #     data.append((fileName, self.tag2label[result[0][1]]))
        for dirName, _, fileNames in os.walk(os.path.join(path)):
            print(dirName)
            tag = os.path.split(dirName)[-1]
            if tag == '.ipynb_checkpoints':
                continue
            fLoop = tqdm(fileNames, desc=dirName)
            for fileName in fLoop:
                if os.path.splitext(fileName)[-1].lower() in self.sufList:
                    data.append((fileName, os.path.join(dirName, fileName), self.tag2label[tag]))
        
        df = pd.DataFrame(data=data, columns = ['file_name', 'file_path', 'label'])
        df.to_csv(os.path.join(path, 'metadata.csv'), index=False)
        return df

from torch.utils.data import Dataset
import pandas as pd
import os
from tqdm import tqdm
import re
from utils import readImage
import numpy as np

class MyDataset(Dataset):
    def __init__(self, datasetPath, dataType='', fileNameForm=None, transform=None, updateIndex=False, tag2label:map=None, tags:list=None, saveConfig=False, reSuffle=False):
        """
        dataType: ['train', 'val', 'test']
        """
        # if dataType:
        #     assert dataType in ['train', 'val', 'test']
        #     self.imageDataPath = os.path.join(datasetPath, dataType, 'images')
        # else:
        #     self.imageDataPath = os.path.join(datasetPath, 'images')
        self.path = datasetPath
        self.imageDataPath = os.path.join(datasetPath, 'images')
        assert os.path.isdir(self.imageDataPath)
        
        self.transform = transform
        self.dataType = dataType
        
        #self.tags = ['tent', 'car', 'truck', 'human', 'bridge', 'bg']
        self.tags =   ['car', 'truck', 'tank', 'armored_car', 'radar', 'artillery', 'person', 'bridge', 'building', 'airport']
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
            self.dataIndex = self.makeIndexFile(self.path, self.dataType)  # update data index
        else:
            # assert os.path.isfile(os.path.join(self.imageDataPath, 'index.csv'))
            # self.dataIndex = pd.read_csv(os.path.join(self.imageDataPath, 'index.csv')) # read data index file
            if  os.path.isfile(os.path.join(self.path, self.dataType + '_metadata.csv')):
                self.dataIndex = pd.read_csv(os.path.join(self.path, self.dataType + '_metadata.csv')) # read data index file
            else:
                self.dataIndex = self.makeIndexFile(self.path, self.dataType)  # update data index

            
            
    def __getitem__(self, index):
        # fileName, tag = 
        row = self.dataIndex.iloc[index]
        fileName, imageFilePath, label = row
        # img = cv2.imread(item.split(' _')[0])
        # imageFilePath = os.path.join(self.imageDataPath, fileName)
        img = None
        try:
            img = readImage(imageFilePath)
            if self.transform:
                img = self.transform(img, return_tensors="pt")
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
    def makeIndexFile(self, path:str, dataType:str='train'):
        data = []
        imgPath = os.path.join(path, 'images')
        # fineNamePattern = re.compile(self.fileNameForm)
        
        # os.system('rm -rf %s'%os.path.join(path,'.ipynb_checkpoints'))

        # for fileName in tqdm(os.listdir(path)):
            # result = re.findall(fineNamePattern, fileName)
            # if result:
            #     # data.append((fileName, result[0][0],  result[0][1]))
            #     data.append((fileName, self.tag2label[result[0][1]]))
        # for dirName, _, fileNames in os.walk(os.path.join(path)):
        #     print(dirName)
        #     tag = os.path.split(dirName)[-1]
        #     if tag == '.ipynb_checkpoints':
        #         continue
        #     fLoop = tqdm(fileNames, desc=dirName)
        #     for fileName in fLoop:
        #         if os.path.splitext(fileName)[-1].lower() in self.sufList:
        #             data.append((fileName, os.path.join(dirName, fileName), self.tag2label[tag]))
        
        # df = pd.DataFrame(data=data, columns = ['file_name', 'file_path', 'label'])
        # df.to_csv(os.path.join(path, 'metadata.csv'), index=False)
        # return df
        data_df = pd.read_csv(os.path.join(path, dataType+'.csv'))
        for index, row in data_df.iterrows():
            data.append((row[1], os.path.abspath(os.path.join(imgPath, row[0], row[1])), self.tag2label[row[0]]))
        
        df = pd.DataFrame(data=data, columns = ['file_name', 'file_path', 'label'])
        
        df.to_csv(os.path.join(path, dataType + '_metadata.csv'), index=False)
        return df

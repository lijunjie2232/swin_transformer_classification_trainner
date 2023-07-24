import os
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import argparse
from multiprocessing import Pool
import imghdr

TAGS = ['car', 'truck', 'tank', 'armored_car', 'radar', 'artillery', 'person', 'bridge', 'building', 'airport', 'bg']
TAG2LABEL = {TAGS[i]:i for i in range(len(TAGS))}
DATA_PATH = './'
TMP_DATAFRAME = None
CHECK_IMG = True


def make_index(tmp_df, imgPath):
    tmp_data = []
    for _, row in tqdm(tmp_df.iterrows(), total=tmp_df.shape[0]):
        path = os.path.abspath(os.path.join(imgPath, row[0], row[1]))
        if CHECK_IMG and not imghdr.what(path):
            continue
        tmp_data.append((row[1], path, TAG2LABEL[row[0]]))
    return pd.DataFrame(data=tmp_data, columns = ['file_name', 'file_path', 'label'])

def makeDataset(path='./', train:int=8, test:int=2, val:int=0, makeIndex=False):
    if not os.path.isdir(os.path.join(path, 'images')):
        print("invalid path")
    imgPath = os.path.join(path, 'images')
    print('form dataset devide method')
    dirSelection = []
    for i in range(train):
        dirSelection.append('train')
    for i in range(test):
        dirSelection.append('test')
    for i in range(val):
        dirSelection.append('val')

    if len(dirSelection) == 0:
        print('invalid dataset split setting')
        return
    dataSet = []
    trainSet = []
    testSet = []
    valSet = []

    for dirName, _, fileNames in tqdm(os.walk(imgPath), desc=imgPath):
        # print(dirName)
        tag = os.path.split(dirName)[-1]
        if tag == '.ipynb_checkpoints':
            continue
        fLoop = tqdm(fileNames, desc=dirName, leave=False)
        for file in fLoop:
            try:
                data = (tag, file)
                dataSet.append(data)
                c = random.choice(dirSelection)
                if c == 'train':
                    trainSet.append(data)
                elif c == 'test':
                    testSet.append(data)
                elif c == 'val':
                    valSet.append(data)
            except Exception as e:
                print(e)
                exit()
    print('dataset: %d'%len(dataSet))
    pd.DataFrame(data=dataSet, columns = ['tag', 'file_name']).to_csv(os.path.join(path, 'dataset.csv'), index=False)
    if trainSet:
        print('train: %d'%len(trainSet))
        tmp_df = pd.DataFrame(data=trainSet, columns = ['tag', 'file_name'])
        tmp_df.to_csv(os.path.join(path, 'train.csv'), index=False)
        if makeIndex:
            print('making index of train ...')
            tmp_df = make_index(tmp_df, imgPath)
            tmp_df.to_csv(os.path.join(path, 'train_metadata.csv'), index=False)
    if testSet:
        print('test: %d'%len(testSet))
        tmp_df = pd.DataFrame(data=testSet, columns = ['tag', 'file_name'])
        tmp_df.to_csv(os.path.join(path, 'test.csv'), index=False)
        if makeIndex:
            print('making index of test ...')
            tmp_df = make_index(tmp_df, imgPath)
            tmp_df.to_csv(os.path.join(path, 'test_metadata.csv'), index=False)
    if valSet:
        print('val: %d'%len(valSet))
        tmp_df = pd.DataFrame(data=valSet, columns = ['tag', 'file_name'])
        tmp_df.to_csv(os.path.join(path, 'val.csv'), index=False)
        if makeIndex:
            print('making index of val ...')
            tmp_df = make_index(tmp_df, imgPath)
            tmp_df.to_csv(os.path.join(path, 'val_metadata.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="split dataset into train|test|val")
    parser.add_argument('--data_path', type=str, default='./', help="path of dataset, which includes images folder")
    parser.add_argument('--train', type=int, default=8, help="train のパーセント")
    parser.add_argument('--test', type=int, default=2, help="test のパ＿セント")
    parser.add_argument('--val', type=int, default=0, help="val のパーセント")
    parser.add_argument('--make_index', default=False, action='store_true', help="make index of absolute path for training")
    parser.add_argument('--check_img', default=False, action='store_true', help="check if image is corrupt")
    args = parser.parse_args()
    DATA_PATH = args.data_path
    CHECK_IMG = args.check_img
    makeDataset(args.data_path, args.train, args.test, args.val, args.make_index)


import os
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import argparse
from multiprocessing import Pool

TAGS = ['car', 'truck', 'tank', 'armored_car', 'radar', 'artillery', 'person', 'bridge', 'building', 'airport']
TAG2LABEL = {TAGS[i]:i for i in range(len(TAGS))}
DATA_PATH = './'
TMP_DATAFRAME = None

def make_row(imgPath, row):
    return (row[1], os.path.abspath(os.path.join(imgPath, row[0], row[1])), TAG2LABEL[row[0]])
# def make_row(imgPath, row_num, tmp_df):
#     row = tmp_df.iloc[row_num]
#     return (row[1], os.path.abspath(os.path.join(imgPath, row[0], row[1])), TAG2LABEL[row[0]])

def make_index(tmp_df, imgPath, p=4):
    print('processes: ', p)
    pool = Pool(p)
    # global TMP_DATAFRAME
    # TMP_DATAFRAME = tmp_df
    # tmp_df = np.array(tmp_df).tolist()
    tmp_data = pool.starmap(make_row, [(imgPath, row) for _, row in tmp_df.iterrows()])
    # tmp_data = pool.starmap(make_row, [(imgPath, row_num, tmp_df) for row_num in range(0, tmp_df.shape[0])])
    # tmp_data = pool.starmap(make_row, [(imgPath, row) for row in tmp_df])
    return pd.DataFrame(data=tmp_data, columns = ['file_name', 'file_path', 'label'])

def makeDataset(path='./', train:int=8, test:int=2, val:int=0, makeIndex=False, p=4):
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

    for dirName, _, fileNames in os.walk(imgPath):
        print(dirName)
        tag = os.path.split(dirName)[-1]
        if tag == '.ipynb_checkpoints':
            continue
        fLoop = tqdm(fileNames, desc=dirName)
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
            tmp_df = make_index(tmp_df, imgPath, p)
            tmp_df.to_csv(os.path.join(path, 'train_metadata.csv'), index=False)
    if testSet:
        print('test: %d'%len(testSet))
        tmp_df = pd.DataFrame(data=testSet, columns = ['tag', 'file_name'])
        tmp_df.to_csv(os.path.join(path, 'test.csv'), index=False)
        if makeIndex:
            print('making index of test ...')
            tmp_df = make_index(tmp_df, imgPath, p)
            tmp_df.to_csv(os.path.join(path, 'test_metadata.csv'), index=False)
    if valSet:
        print('val: %d'%len(valSet))
        tmp_df = pd.DataFrame(data=valSet, columns = ['tag', 'file_name'])
        tmp_df.to_csv(os.path.join(path, 'val.csv'), index=False)
        if makeIndex:
            print('making index of val ...')
            tmp_df = make_index(tmp_df, imgPath, p)
            tmp_df.to_csv(os.path.join(path, 'val_metadata.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="split dataset into train|test|val")
    parser.add_argument('--data_path', type=str, default='./', help="path of dataset, which includes images folder")
    parser.add_argument('--train', type=int, default=8, help="train のパーセント")
    parser.add_argument('--test', type=int, default=2, help="test のパ＿セント")
    parser.add_argument('--val', type=int, default=0, help="val のパーセント")
    parser.add_argument('--make_index', default=False, action='store_true', help="make index of absolute path for training")
    parser.add_argument('--p', type=int, default=4, help="parr_num")
    args = parser.parse_args()
    DATA_PATH = args.data_path
    makeDataset(args.data_path, args.train, args.test, args.val, args.make_index, args.p)


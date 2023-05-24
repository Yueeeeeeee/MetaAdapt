import os
from pathlib import Path
from config import args

import json
import random
from abc import *
import numpy as np
import pandas as pd

from collections import defaultdict


def get_dataset(args, mode, dataset, path):
    assert dataset.lower() in ['constraint', 'coaid', 'fakecovid', 'antivax', 'liar', 'fever',
        'fakenews', 'gettingreal', 'pheme', 'health', 'gossipcop', 'politifact', 'multisource']
    assert mode in ['train', 'val', 'test']

    if dataset.lower() == 'constraint':
        return ConstraintDataset(args, mode, path)
    elif dataset.lower() == 'coaid':
        return CoAIDDataset(args, mode, path)
    elif dataset.lower() == 'fakecovid':
        return FakeCovidDataset(args, mode, path)
    elif dataset.lower() == 'antivax':
        return ANTiVaxDataset(args, mode, path)
    elif dataset.lower() == 'liar':
        return LIARDataset(args, mode, path)
    elif dataset.lower() == 'fever':
        return FEVERDataset(args, mode, path)
    elif dataset.lower() == 'fakenews':
        return FakeNewsDataset(args, mode, path)
    elif dataset.lower() == 'gettingreal':
        return GettingRealDataset(args, mode, path)
    elif dataset.lower() == 'pheme':
        return PHEMEDataset(args, mode, path)
    elif dataset.lower() == 'health':
        return HealthDataset(args, mode, path)
    elif dataset.lower() == 'gossipcop':
        return GossipCopDataset(args, mode, path)
    elif dataset.lower() == 'politifact':
        return PolitiFactDataset(args, mode, path)
    elif dataset.lower() == 'multisource':
        return MultisourceDataset(args, mode, path)
    

class AbstractDataset(metaclass=ABCMeta):
    def __init__(self, args, mode, path):
        self.args = args
        self.mode = mode
        self.path = path

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @abstractmethod
    def load_dataset(self):
        pass

    def get_train_val_test_split_path(self):
        return os.path.join(self.path,
            self.code()+'_'+str(self.args.val_size)+'_'+str(self.args.test_size)+'_split.json')

    def get_train_val_test_split(self):
        split_path = self.get_train_val_test_split_path()
        return json.loads(open(split_path).read())

    def make_train_val_test_split(self, length, val=None, test=None):
        if val is None:
            val = self.args.val_size
        if test is None:
            test = self.args.test_size
        
        split_path = self.get_train_val_test_split_path()
        indices = list(range(length))
        random.shuffle(indices)
        split = {
            'train': indices[:int(length*(1-val-test))],
            'val': indices[int(length*(1-val-test)):int(length*(1-test))],
            'test': indices[int(length*(1-test)):]
        }
        with open(split_path, 'w') as f:
            json.dump(split, f)
        return split


class ConstraintDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'constraint'

    def load_dataset(self):
        if self.mode in ['train', 'val']:
            file_path = os.path.join(self.path, 'Constraint_Train.csv')
        elif self.mode == 'test':
            file_path = os.path.join(self.path, 'Constraint_Val.csv')
        
        raw_data = pd.read_csv(file_path)
        raw_data["label"] = raw_data["label"].map({"real": 1, "fake": 0})
        data = raw_data.drop(["id"], axis=1).values

        if self.mode == 'test':
            return data[:, 0], data[:, 1].astype(int)
        else:
            split_file = self.get_train_val_test_split_path()
            if os.path.isfile(split_file):
                split = self.get_train_val_test_split()
            else:
                split = self.make_train_val_test_split(len(data), test=0)
            indices = split[self.mode]

            return data[indices, 0], data[indices, 1].astype(int)


class CoAIDDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'coaid'
    
    def load_dataset(self):
        dates = ['05-01-2020', '07-01-2020', '09-01-2020', '11-01-2020']

        data = []
        labels = []
        for date in dates:
            news_path = os.path.join(self.path, date, 'NewsRealCOVID-19.csv')
            claim_path = os.path.join(self.path, date, 'ClaimRealCOVID-19.csv')
            raw_data = pd.read_csv(news_path)[['content']].values.squeeze().tolist()
            if os.path.isfile(claim_path):
                raw_data += pd.read_csv(claim_path)[['title']].values.squeeze().tolist()
            for entry in raw_data:
                if isinstance(entry, str):
                    data.append(entry)
                    labels.append(1)

        for date in dates:
            news_path = os.path.join(self.path, date, 'NewsFakeCOVID-19.csv')
            claim_path = os.path.join(self.path, date, 'ClaimFakeCOVID-19.csv')
            raw_data = pd.read_csv(news_path)[['content']].values.squeeze().tolist()
            if os.path.isfile(claim_path):
                raw_data += pd.read_csv(claim_path)[['title']].values.squeeze().tolist()
            for entry in raw_data:
                if isinstance(entry, str):
                    data.append(entry)
                    labels.append(0)
        
        split_file = self.get_train_val_test_split_path()
        if os.path.isfile(split_file):
            split = self.get_train_val_test_split()
        else:
            split = self.make_train_val_test_split(len(data))
        indices = split[self.mode]
        
        return np.array(data)[indices], np.array(labels)[indices]


class FakeCovidDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'fakecovid'
    
    def load_dataset(self):
        label_mapping = defaultdict(lambda: 1)
        fake_labels = ['false', 'mostly false', 'pants on fire', 'pants on fire!',
            'incorrect', 'fake', 'mosly false', 'three pinocchios', 'four pinocchios', 
            'not true', 'scam', 'conspiracy theory']
        for label in fake_labels:
            label_mapping[label] = 0

        data = []
        labels = []
        
        raw_data = pd.read_csv(os.path.join(self.path, 'data', 'FakeCovid_June2020.csv'))
        raw_data = raw_data[raw_data['lang']=='English']
        raw_data = raw_data[['content_text', 'class']]
        raw_data['class'] = raw_data['class'].str.lower()
        raw_data['class'] = raw_data['class'].map(label_mapping)
        raw_data = raw_data.values
        for i in range(len(raw_data)):
            data.append(raw_data[i, 0])
            labels.append(raw_data[i, 1])

        raw_data = pd.read_csv(os.path.join(self.path, 'data', 'FakeCovid_July2020.csv'))
        raw_data = raw_data[raw_data['lang']=='en']
        raw_data = raw_data[['content_text', 'class']]
        raw_data['class'] = raw_data['class'].str.lower()
        raw_data['class'] = raw_data['class'].map(label_mapping)
        raw_data = raw_data.values
        for i in range(len(raw_data)):
            data.append(raw_data[i, 0])
            labels.append(raw_data[i, 1])
        
        split_file = self.get_train_val_test_split_path()
        if os.path.isfile(split_file):
            split = self.get_train_val_test_split()
        else:
            split = self.make_train_val_test_split(len(data))
        indices = split[self.mode]
        
        return np.array(data)[indices], np.array(labels)[indices]


class ANTiVaxDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'antivax'

    def load_dataset(self):
        file_path = os.path.join(self.path, 'dataset.csv')
        data = pd.read_csv(file_path).dropna().values

        split_file = self.get_train_val_test_split_path()
        if os.path.isfile(split_file):
            split = self.get_train_val_test_split()
        else:
            split = self.make_train_val_test_split(len(data))
        indices = split[self.mode]

        return data[indices, 0], data[indices, 1].astype(int)


class LIARDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'liar'
    
    def load_dataset(self):
        label_mapping = defaultdict(lambda: 1)
        fake_labels = ['false', 'barely-true', 'pants-fire']
        for label in fake_labels:
            label_mapping[label] = 0

        if self.mode == 'train':
            file_path = os.path.join(self.path, 'train.tsv')
        elif self.mode == 'val':
            file_path = os.path.join(self.path, 'valid.tsv')
        elif self.mode == 'test':
            file_path = os.path.join(self.path, 'test.tsv')
        
        raw_data = pd.read_csv(file_path, sep='\t', header=None)
        raw_data = raw_data.iloc[:, 1:3]
        raw_data.columns = ['label', 'text']
        raw_data['label'] = raw_data["label"].map(label_mapping)
        data = raw_data.values

        return data[:, 1], data[:, 0].astype(int)


class FEVERDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'fever'
    
    def load_dataset(self):
        if self.mode == 'train':
            file_path = os.path.join(self.path, 'train.jsonl')
        elif self.mode == 'val':
            file_path = os.path.join(self.path, 'paper_dev.jsonl')
        elif self.mode == 'test':
            file_path = os.path.join(self.path, 'paper_test.jsonl')

        data = []
        labels = []
        
        for line in open(file_path).readlines():
            cur_file = json.loads(line)
            if cur_file['verifiable'] == 'VERIFIABLE':
                if cur_file['label'] == 'REFUTES':
                    data.append(cur_file['claim'])
                    labels.append(0)
                elif cur_file['label'] == 'SUPPORTS':
                    data.append(cur_file['claim'])
                    labels.append(1)

        return np.array(data), np.array(labels)


class FakeNewsDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'fakenews'
    
    def load_dataset(self):
        data = []
        labels = []
        
        raw_data = pd.read_csv(os.path.join(self.path, 'train.csv'))
        raw_data = raw_data[['text', 'label']]
        raw_data['label'] = raw_data['label'].map({0: 1, 1: 0})
        raw_data = raw_data.values
        for i in range(len(raw_data)):
            if isinstance(raw_data[i, 0], str):
                data.append(raw_data[i, 0])
                labels.append(raw_data[i, 1])
        
        split_file = self.get_train_val_test_split_path()
        if os.path.isfile(split_file):
            split = self.get_train_val_test_split()
        else:
            split = self.make_train_val_test_split(len(data))
        indices = split[self.mode]
        
        return np.array(data)[indices], np.array(labels)[indices]


class GettingRealDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'gettingreal'
    
    def load_dataset(self):
        label_mapping = defaultdict(lambda: 1)
        fake_labels = ['bias', 'conspiracy', 'fake', 'hate']
        for label in fake_labels:
            label_mapping[label] = 0

        data = []
        labels = []
        
        raw_data = pd.read_csv(os.path.join(self.path, 'fake.csv'))
        raw_data = raw_data[['text', 'type']]
        raw_data['type'] = raw_data['type'].str.lower()
        raw_data['type'] = raw_data['type'].map(label_mapping)
        raw_data = raw_data.values

        for i in range(len(raw_data)):
            if isinstance(raw_data[i, 0], str):
                data.append(raw_data[i, 0])
                labels.append(raw_data[i, 1])
        
        split_file = self.get_train_val_test_split_path()
        if os.path.isfile(split_file):
            split = self.get_train_val_test_split()
        else:
            split = self.make_train_val_test_split(len(data))
        indices = split[self.mode]
        
        return np.array(data)[indices], np.array(labels)[indices]


class PHEMEDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'pheme'
    
    def load_dataset(self):
        folders = ['charliehebdo', 'ferguson', 'germanwings-crash', 'ottawashooting', 'sydneysiege']
        
        data = []
        labels = []
        for f in folders:
            real_paths = os.listdir(os.path.join(self.path, f, 'non-rumours'))
            fake_paths = os.listdir(os.path.join(self.path, f, 'rumours'))
            
            for path in real_paths:
                cur_path = os.path.join(self.path, f, 'non-rumours', path, 'source-tweet')
                try:
                    f_path = os.path.join(cur_path, os.listdir(cur_path)[0])
                    text = json.loads(open(f_path).read())['text']
                except:
                    continue
                data.append(text)
                labels.append(1)

            for path in fake_paths:
                cur_path = os.path.join(self.path, f, 'rumours', path, 'source-tweet')
                try:
                    f_path = os.path.join(cur_path, os.listdir(cur_path)[0])
                    text = json.loads(open(f_path).read())['text']
                except:
                    continue
                data.append(text)
                labels.append(0)
        
        split_file = self.get_train_val_test_split_path()
        if os.path.isfile(split_file):
            split = self.get_train_val_test_split()
        else:
            split = self.make_train_val_test_split(len(data))
        indices = split[self.mode]
        
        return np.array(data)[indices], np.array(labels)[indices]


class HealthDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'health'
    
    def load_dataset(self):
        real_files = ['RealCancer_cleaned.csv', 'RealDiabetes_cleaned.csv']
        fake_files = ['FakeCancer_cleaned.csv', 'FakeDiabetes_cleaned.csv']
        
        data = []
        labels = []  
        for f in real_files:
            raw_data = pd.read_csv(os.path.join(self.path, f))
            raw_data = raw_data['content'].values
            for i in range(len(raw_data)):
                if isinstance(raw_data[i], str):
                    data.append(raw_data[i])
                    labels.append(1)

        for f in fake_files:
            raw_data = pd.read_csv(os.path.join(self.path, f))
            raw_data = raw_data['content'].values
            for i in range(len(raw_data)):
                if isinstance(raw_data[i], str):
                    data.append(raw_data[i])
                    labels.append(0)

        split_file = self.get_train_val_test_split_path()
        if os.path.isfile(split_file):
            split = self.get_train_val_test_split()
        else:
            split = self.make_train_val_test_split(len(data))
        indices = split[self.mode]
        
        return np.array(data)[indices], np.array(labels)[indices]


class GossipCopDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'gossipcop'
    
    def load_dataset(self):
        data = []
        labels = []
        real_paths = os.listdir(os.path.join(self.path, 'gossipcop', 'real'))
        fake_paths = os.listdir(os.path.join(self.path, 'gossipcop', 'fake'))
            
        for path in real_paths:
            cur_path = os.path.join(self.path, 'gossipcop', 'real', path, 'news content.json')
            try:
                text = json.loads(open(cur_path).read())['text']
            except:
                continue
            data.append(text)
            labels.append(1)

        for path in fake_paths:
            cur_path = os.path.join(self.path, 'gossipcop', 'fake', path, 'news content.json')
            try:
                text = json.loads(open(cur_path).read())['text']
            except:
                continue
            data.append(text)
            labels.append(0)
        
        split_file = self.get_train_val_test_split_path()
        if os.path.isfile(split_file):
            split = self.get_train_val_test_split()
        else:
            split = self.make_train_val_test_split(len(data))
        indices = split[self.mode]
        
        return np.array(data)[indices], np.array(labels)[indices]


class PolitiFactDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'politifact'
    
    def load_dataset(self):
        data = []
        labels = []
        real_paths = os.listdir(os.path.join(self.path, 'politifact', 'real'))
        fake_paths = os.listdir(os.path.join(self.path, 'politifact', 'fake'))
            
        for path in real_paths:
            cur_path = os.path.join(self.path, 'politifact', 'real', path, 'news content.json')
            try:
                text = json.loads(open(cur_path).read())['text']
            except:
                continue
            data.append(text)
            labels.append(1)

        for path in fake_paths:
            cur_path = os.path.join(self.path, 'politifact', 'fake', path, 'news content.json')
            try:
                text = json.loads(open(cur_path).read())['text']
            except:
                continue
            data.append(text)
            labels.append(0)
        
        split_file = self.get_train_val_test_split_path()
        if os.path.isfile(split_file):
            split = self.get_train_val_test_split()
        else:
            split = self.make_train_val_test_split(len(data))
        indices = split[self.mode]
        
        return np.array(data)[indices], np.array(labels)[indices]


class MultisourceDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'multisource'
    
    def load_dataset(self):
        if self.mode == 'train':
            file_path = os.path.join(self.path, 'train.json')
        elif self.mode == 'val':
            file_path = os.path.join(self.path, 'val.json')
        elif self.mode == 'test':
            file_path = os.path.join(self.path, 'test.json')
        data = json.loads(open(file_path).read())
        
        return np.array(data['data']), np.array(data['labels'])
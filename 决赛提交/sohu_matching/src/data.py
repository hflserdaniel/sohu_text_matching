import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, AutoTokenizer
from utils import pad_to_maxlen, augment_data
import pandas as pd

from tqdm import tqdm
import json

# the main difference between the two datasets is 
# the length limit (512 for one sentence in SBERT
# but for the two concated sentences in BERT setting) 
class SentencePairDatasetForSBERT(Dataset):
    def __init__(self, file_dir, is_train, tokenizer_config, shuffle_order=False, aug_data=False, len_limit=512, clip='head'):
        self.is_train = is_train
        self.shuffle_order = shuffle_order
        self.aug_data = aug_data
        self.total_source_input_ids = []
        # token_types are no longer neccessary if not concat into one text
        # self.total_source_input_types = []
        self.total_target_input_ids = []
        # self.total_target_input_types = []
        self.sample_types = []

        # use AutoTokenzier instead of BertTokenizer to support speice.model (AlbertTokenizer-like)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_config)
        lines = []
        for single_file_dir in file_dir:
            with open(single_file_dir, 'r', encoding='utf-8') as f_in:
                content = f_in.readlines()
                for item in content:
                    line = json.loads(item.strip())
                    if not is_train:
                        line['type'] = 0 if 'a' in line['id'] else 1
                    lines.append(line)

        content = pd.DataFrame(lines)
        content.columns = ['source', 'target', 'label', 'type']

        # utilize labelB=1-->A positive, labelA=0-->B negative 
        if self.is_train and self.aug_data:
            print("augmenting data...")
            content = augment_data(content)

        sources = content['source'].values.tolist()
        targets = content['target'].values.tolist()

        self.sample_types = content['type'].values.tolist()
        if self.is_train:
            self.labels = content['label'].values.tolist()
        else:
            self.ids = content['label'].values.tolist()

        # shuffle_order is only allowed for training mode
        if self.shuffle_order and self.is_train:
            sources += content['target'].values.tolist()
            targets += content['source'].values.tolist()
            self.labels += self.labels
            self.sample_types += self.sample_types

        for source, target in tqdm(zip(sources, targets), total=len(sources)):
            # tokenize before clipping
            source = tokenizer.encode(source)[1:-1]
            target = tokenizer.encode(target)[1:-1]

            # clip the sentences if too long
            # TODO: different strategies to clip long sequences
            if clip == 'head':
                if len(source)+2 > len_limit:
                    source = source[0: len_limit-2]
                if len(target)+2 > len_limit:
                    target = target[0: len_limit-2]

            if clip == 'tail':
                if len(source)+2 > len_limit:
                    source = source[-len_limit+2:]
                if len(target)+2 > len_limit:
                    target = target[-len_limit+2:]

            # check if the length is within the limit
            assert len(source)+2 <= len_limit and len(target)+2 <= len_limit
            
            # [CLS]:101, [SEP]:102
            source_input_ids = [101] + source + [102]
            target_input_ids = [101] + target + [102]

            assert len(source_input_ids) <= len_limit and len(target_input_ids) <= len_limit
            
            self.total_source_input_ids.append(source_input_ids)
            self.total_target_input_ids.append(target_input_ids)
    
        self.max_source_input_len = max([len(s) for s in self.total_source_input_ids])
        self.max_target_input_len = max([len(s) for s in self.total_target_input_ids])
        print("max source length: ", self.max_source_input_len)
        print("max target length: ", self.max_target_input_len)

    def __len__(self):
        return len(self.total_target_input_ids)

    def __getitem__(self, idx):
        source_input_ids = pad_to_maxlen(self.total_source_input_ids[idx], self.max_source_input_len)
        target_input_ids = pad_to_maxlen(self.total_target_input_ids[idx], self.max_target_input_len)
        sample_type = int(self.sample_types[idx])

        if self.is_train:
            label = int(self.labels[idx])
            return torch.LongTensor(source_input_ids), torch.LongTensor(target_input_ids), torch.LongTensor([label]), sample_type
        
        else:
            index = self.ids[idx]
            return torch.LongTensor(source_input_ids), torch.LongTensor(target_input_ids), index, sample_type 

class SentencePairDatasetWithType(Dataset):
    def __init__(self, file_dir, is_train, tokenizer_config, shuffle_order=False, aug_data=False, len_limit=512, clip='head'):
        self.is_train = is_train
        self.shuffle_order = shuffle_order
        self.aug_data = aug_data
        self.total_input_ids = []
        self.total_input_types = []
        self.sample_types = []

        # use AutoTokenzier instead of BertTokenizer to support speice.model (AlbertTokenizer-like)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_config)

        # read json lines and convert to dict / df
        lines = []
        for single_file_dir in file_dir:
            with open(single_file_dir, 'r', encoding='utf-8') as f_in:
                content = f_in.readlines()
                for item in content:
                    line = json.loads(item.strip())

                    # for final stage, task a and b are included in the same file
                    if not is_train:
                        line['type'] = 0 if 'a' in line['id'] else 1
                    lines.append(line)
            print(single_file_dir, len(lines))
        content = pd.DataFrame(lines)
        # print(content.head())
        content.columns = ['source', 'target', 'label', 'type']

        # utilize labelB=1-->A positive, labelA=0-->B negative 
        if self.is_train and self.aug_data:
            print("augmenting data...")
            content = augment_data(content)

        sources = content['source'].values.tolist()
        targets = content['target'].values.tolist()

        self.sample_types = content['type'].values.tolist()
        if self.is_train:
            self.labels = content['label'].values.tolist()
        else:
            self.ids = content['label'].values.tolist()

        # shuffle_order is only allowed for training mode
        if self.shuffle_order and self.is_train:
            sources += content['target'].values.tolist()
            targets += content['source'].values.tolist()
            self.labels += self.labels
            self.sample_types += self.sample_types
            
        len_limit_s = (len_limit-3)//2
        len_limit_t = (len_limit-3)-len_limit_s
        # print('len_limit_s: ', len_limit_s)
        # print('len_limit_t: ', len_limit_t)
        for source, target in tqdm(zip(sources, targets), total=len(sources)):
            # tokenize before clipping
            source = tokenizer.encode(source)[1:-1]
            target = tokenizer.encode(target)[1:-1]

            # clip the sentences if too long
            # TODO: different strategies to clip long sequences
            if clip == 'head' and len(source)+len(target)+3 > len_limit:
                if len(source)>len_limit_s and len(target)>len_limit_t:
                    source = source[0:len_limit_s]
                    target = target[0:len_limit_t]
                elif len(source)>len_limit_s:
                    source = source[0:len_limit-3-len(target)]
                elif len(target)>len_limit_t:
                    target = target[0:len_limit-3-len(source)]
            
            if clip == 'tail' and len(source)+len(target)+3 > len_limit:
                if len(source)>len_limit_s and len(target)>len_limit_t:
                    source = source[-len_limit_s:]
                    target = target[-len_limit_t:]
                elif len(source)>len_limit_s:
                    source = source[-(len_limit-3-len(target)):]
                elif len(target)>len_limit_t:
                    target = target[-(len_limit-3-len(source)):]

            # check if the total length is within the limit
            assert len(source)+len(target)+3 <= len_limit
            
            # [CLS]:101, [SEP]:102
            input_ids = [101] + source + [102] + target + [102]
            input_types = [0]*(len(source)+2) + [1]*(len(target)+1)

            assert len(input_ids) <= len_limit and len(input_types) <= len_limit
            self.total_input_ids.append(input_ids)
            self.total_input_types.append(input_types)
    
        self.max_input_len = max([len(s) for s in self.total_input_ids])
        print("max length: ", self.max_input_len)

    def __len__(self):
        return len(self.total_input_ids)

    def __getitem__(self, idx):
        if self.is_train:
            input_ids = pad_to_maxlen(self.total_input_ids[idx], self.max_input_len)
            input_types = pad_to_maxlen(self.total_input_types[idx], self.max_input_len)
            label = int(self.labels[idx])
            sample_type = int(self.sample_types[idx])
            # print(len(input_ids), len(input_types), label)
            return torch.LongTensor(input_ids), torch.LongTensor(input_types), torch.LongTensor([label]), sample_type
            
        else:
            input_ids = pad_to_maxlen(self.total_input_ids[idx], self.max_input_len)
            input_types = pad_to_maxlen(self.total_input_types[idx], self.max_input_len)
            index  = self.ids[idx]
            sample_type = int(self.sample_types[idx])
            return torch.LongTensor(input_ids), torch.LongTensor(input_types), index, sample_type

# NOT CURRENTLY IN USE
# template for the dataset of multiple task types
# compatible with training code by changing task_num 
class SentencePairDatasetWithMultiType(Dataset):
    def __init__(self, file_dir, is_train, tokenizer_config, shuffle_order=False, aug_data=False, len_limit=512, clip='head'):
        self.is_train = is_train
        self.shuffle_order = shuffle_order
        self.aug_data = aug_data
        self.total_input_ids = []
        self.total_input_types = []
        self.sample_types = []

        # use AutoTokenzier instead of BertTokenizer to support speice.model (AlbertTokenizer-like)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_config)

        # read json lines and convert to dict / df
        lines = []
        for single_file_dir in file_dir:
            with open(single_file_dir, 'r', encoding='utf-8') as f_in:
                content = f_in.readlines()
                for item in content:
                    line = json.loads(item.strip())
                    # BUG FIXED, order MATTERS!
                    # mannually add key 'type' to distinguish the origin of samples
                    # 0 for A, 1 for B
                    if 'A' in single_file_dir:
                        if self.is_train:
                            line['label'] = line.pop('labelA')   
                        # assign type according to task names
                        if '短短' in single_file_dir:
                            line['type'] = 0
                        elif '短长' in single_file_dir:
                            line['type'] = 2
                        else:
                            line['type'] = 4
                    else:
                        if self.is_train:
                            line['label'] = line.pop('labelB')   
                        # assign type according to task names
                        if '短短' in single_file_dir:
                            line['type'] = 1
                        elif '短长' in single_file_dir:
                            line['type'] = 3
                        else:
                            line['type'] = 5
                    lines.append(line)
            print(single_file_dir, len(lines))
        content = pd.DataFrame(lines)
        # print(content.head())
        content.columns = ['source', 'target', 'label', 'type']

        # utilize labelB=1-->A positive, labelA=0-->B negative 
        if self.is_train and self.aug_data:
            print("augmenting data...")
            content = augment_data(content)

        sources = content['source'].values.tolist()
        targets = content['target'].values.tolist()

        self.sample_types = content['type'].values.tolist()
        if self.is_train:
            self.labels = content['label'].values.tolist()
        else:
            self.ids = content['label'].values.tolist()

        # shuffle_order is only allowed for training mode
        if self.shuffle_order and self.is_train:
            sources += content['target'].values.tolist()
            targets += content['source'].values.tolist()
            self.labels += self.labels
            self.sample_types += self.sample_types
            
        len_limit_s = (len_limit-3)//2
        len_limit_t = (len_limit-3)-len_limit_s
        # print('len_limit_s: ', len_limit_s)
        # print('len_limit_t: ', len_limit_t)
        for source, target in tqdm(zip(sources, targets), total=len(sources)):
            # tokenize before clipping
            source = tokenizer.encode(source)[1:-1]
            target = tokenizer.encode(target)[1:-1]

            # clip the sentences if too long
            # TODO: different strategies to clip long sequences
            if clip == 'head' and len(source)+len(target)+3 > len_limit:
                if len(source)>len_limit_s and len(target)>len_limit_t:
                    source = source[0:len_limit_s]
                    target = target[0:len_limit_t]
                elif len(source)>len_limit_s:
                    source = source[0:len_limit-3-len(target)]
                elif len(target)>len_limit_t:
                    target = target[0:len_limit-3-len(source)]
            
            if clip == 'tail' and len(source)+len(target)+3 > len_limit:
                if len(source)>len_limit_s and len(target)>len_limit_t:
                    source = source[-len_limit_s:]
                    target = target[-len_limit_t:]
                elif len(source)>len_limit_s:
                    source = source[-(len_limit-3-len(target)):]
                elif len(target)>len_limit_t:
                    target = target[-(len_limit-3-len(source)):]

            # check if the total length is within the limit
            assert len(source)+len(target)+3 <= len_limit
            
            # [CLS]:101, [SEP]:102
            input_ids = [101] + source + [102] + target + [102]
            input_types = [0]*(len(source)+2) + [1]*(len(target)+1)

            assert len(input_ids) <= len_limit and len(input_types) <= len_limit
            self.total_input_ids.append(input_ids)
            self.total_input_types.append(input_types)
    
        self.max_input_len = max([len(s) for s in self.total_input_ids])
        print("max length: ", self.max_input_len)

    def __len__(self):
        return len(self.total_input_ids)

    def __getitem__(self, idx):
        if self.is_train:
            input_ids = pad_to_maxlen(self.total_input_ids[idx], self.max_input_len)
            input_types = pad_to_maxlen(self.total_input_types[idx], self.max_input_len)
            label = int(self.labels[idx])
            sample_type = int(self.sample_types[idx])
            # print(len(input_ids), len(input_types), label)
            return torch.LongTensor(input_ids), torch.LongTensor(input_types), torch.LongTensor([label]), sample_type
            
        else:
            input_ids = pad_to_maxlen(self.total_input_ids[idx], self.max_input_len)
            input_types = pad_to_maxlen(self.total_input_types[idx], self.max_input_len)
            index  = self.ids[idx]
            sample_type = int(self.sample_types[idx])
            return torch.LongTensor(input_ids), torch.LongTensor(input_types), index, sample_type
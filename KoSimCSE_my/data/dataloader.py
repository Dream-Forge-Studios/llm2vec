import numpy
import torch
import logging
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset, DatasetDict

logger = logging.getLogger(__name__)


class ModelDataLoader(Dataset):
    def __init__(self, file_path, args, metric, tokenizer, type_):
        self.type = type_
        self.args = args
        self.metric = metric

        """NLI"""
        self.anchor = []
        self.positive = []
        self.negative = []

        """STS"""
        self.label = []
        self.sentence_1 = []
        self.sentence_2 = []

        #  -------------------------------------
        self.bert_tokenizer = tokenizer
        self.file_path = file_path

        """
        [CLS]: 2
        [PAD]: 0
        [UNK]: 1
        """
        self.init_token = self.bert_tokenizer.cls_token
        self.pad_token = self.bert_tokenizer.pad_token
        self.unk_token = self.bert_tokenizer.unk_token

        self.init_token_idx = self.bert_tokenizer.convert_tokens_to_ids(self.init_token)
        self.pad_token_idx = self.bert_tokenizer.convert_tokens_to_ids(self.pad_token)
        self.unk_token_idx = self.bert_tokenizer.convert_tokens_to_ids(self.unk_token)
        
    def load_data(self, type):
        datas = self.file_path
        for i, data in enumerate(datas):
            if i + 1 == len(datas):
                self.data2tensor(data, datas[0], type)
            else:
                self.data2tensor(data, datas[i + 1], type)

        if type == 'train':
            # assert len(self.anchor) == len(self.positive)
            assert len(self.anchor) == len(self.positive) == len(self.negative)
        else:
            assert len(self.sentence_1) == len(self.sentence_2) == len(self.label)

    def data2tensor(self, data, neg_data, type):
        if type == 'train':
            anchor_sen = data['instruction']
            positive_sen = data['output']
            negative_sen = neg_data['output']

            anchor = self.bert_tokenizer(anchor_sen, 
                                         truncation=True,
                                         return_tensors="pt",
                                         max_length=self.args.max_len,
                                         pad_to_max_length="right")
            
            positive = self.bert_tokenizer(positive_sen, 
                                           truncation=True,
                                           return_tensors="pt",
                                           max_length=self.args.max_len,
                                           pad_to_max_length="right")

            negative = self.bert_tokenizer(negative_sen,
                                           truncation=True,
                                           return_tensors="pt",
                                           max_length=self.args.max_len,
                                           pad_to_max_length="right")
            
            self.anchor.append(anchor)
            self.positive.append(positive)
            self.negative.append(negative)

        # else:
        #     sentence_1, sentence_2, label = split_data
        #
        #     sentence_1 = self.bert_tokenizer(sentence_1,
        #                                      truncation=True,
        #                                      return_tensors="pt",
        #                                      max_length=self.args.max_len,
        #                                      pad_to_max_length="right")
        #
        #     sentence_2 = self.bert_tokenizer(sentence_2,
        #                                      truncation=True,
        #                                      return_tensors="pt",
        #                                      max_length=self.args.max_len,
        #                                      pad_to_max_length="right")
        #
        #
        #     self.sentence_1.append(sentence_1)
        #     self.sentence_2.append(sentence_2)
        #     self.label.append(float(label.strip())/5.0)

    def __getitem__(self, index):

        if self.type == 'train':
            inputs = {'anchor': {
                'source': torch.LongTensor(self.anchor[index]['input_ids']),
                'attention_mask': self.anchor[index]['attention_mask'],
                'token_type_ids': torch.LongTensor(self.anchor[index]['token_type_ids'])
                },
                      'positive': {
                'source': torch.LongTensor(self.positive[index]['input_ids']),
                'attention_mask': self.positive[index]['attention_mask'],
                'token_type_ids': torch.LongTensor(self.positive[index]['token_type_ids'])
                },
                      'negative': {
                'source': torch.LongTensor(self.negative[index]['input_ids']),
                'attention_mask': self.negative[index]['attention_mask'],
                'token_type_ids': torch.LongTensor(self.negative[index]['token_type_ids'])
                }
            }
        else:

            inputs = {'sentence_1': {
                'source': torch.LongTensor(self.sentence_1[index]['input_ids']),
                'attention_mask': self.sentence_1[index]['attention_mask'],
                'token_type_ids': torch.LongTensor(self.sentence_1[index]['token_type_ids'])
                },
                      'sentence_2': {
                'source': torch.LongTensor(self.sentence_2[index]['input_ids']),
                'attention_mask': self.sentence_2[index]['attention_mask'],
                'token_type_ids': torch.LongTensor(self.sentence_2[index]['token_type_ids'])
                },
                      'label': {
                          'value': torch.FloatTensor([self.label[index]])}
                }

        for key, value in inputs.items():
            for inner_key, inner_value in value.items():
                inputs[key][inner_key] = inner_value.squeeze(0)
                
        inputs = self.metric.move2device(inputs, self.args.device)
        
        return inputs

    def __len__(self):
        if self.type == 'train':
            return len(self.anchor)
        else:
            return len(self.label)


# Get train, valid, test data loader and BERT tokenizer
def get_loader(args, metric):
    file_path = "maywell/ko_wikidata_QA"
    cache_dir = "/data/llm/"
    cache_dir = "D:\\huggingface\\cache"

    raw_datasets = load_dataset(file_path, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    train_testsplit = raw_datasets['train'].train_test_split(test_size=0.1, seed=42)
    fix_datasets = DatasetDict({
        # 'train': train_testsplit['train'].select(range(10)),
        'train': train_testsplit['train'],
        'validation': train_testsplit['test']
    })

    if args.train == 'True' and args.test == 'False':
        train_iter = ModelDataLoader(fix_datasets['train'], args, metric, tokenizer, type_='train')
        # valid_iter = ModelDataLoader(file_path, args, metric, tokenizer, type_='valid')

        # valid_iter.load_data('valid')
        train_iter.load_data('train')

        loader = {'train': DataLoader(dataset=train_iter,
                                      batch_size=args.batch_size,
                                      shuffle=True),
                  # 'valid': DataLoader(dataset=valid_iter,
                  #                     batch_size=args.batch_size,
                  #                     shuffle=True)
                  }
    else:
        loader = None

    return loader, tokenizer


def convert_to_tensor(corpus, tokenizer, device):
    inputs = tokenizer(corpus,
                       truncation=True,
                       return_tensors="pt",
                       max_length=50,
                       pad_to_max_length="right")
    
    embedding = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    token_type_ids = inputs['token_type_ids']
        
    inputs = {'source': torch.LongTensor(embedding).to(device),
              'token_type_ids': torch.LongTensor(token_type_ids).to(device),
              'attention_mask': attention_mask.to(device)}
    
    return inputs


def example_model_setting(model_ckpt, model_name):

    from model.simcse.bert import BERT

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = BERT(AutoModel.from_pretrained(model_name))
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model.load_state_dict(torch.load(model_ckpt)['model'])
    model.to(device)
    model.eval()
    
    return model, tokenizer, device


if __name__ == '__main__':
    get_loader('test')

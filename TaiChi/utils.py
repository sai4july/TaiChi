import torch
import warnings
from args import *
import pandas as pd
import torch.nn as nn
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import torch.nn.functional as F
import random

warnings.filterwarnings('ignore')

parser = get_parser()
args = parser.parse_args()


def load_data(dataset=None):

    print(f'__loading__{args.dataset}__')
    train_df = pd.read_csv(f"datasets/{args.dataset}/train.tsv",'\t')[:10000]
    dev_df = pd.read_csv(f"datasets/{args.dataset}/dev.tsv",'\t')
    test_df = pd.read_csv(f"datasets/{args.dataset}/test.tsv",'\t')
    adv_df = pd.read_csv(f"adv_datasets/{args.task}_{args.adv_attack}/train.tsv",'\t')
    attack_df = pd.read_csv(f"attack_dataset/{args.task}_{args.attack}_test.tsv",'\t')
    return train_df,dev_df,test_df,attack_df,adv_df


class Bert_dataset(Dataset):
    def __init__(self,df):
        self.df=df
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_type,do_lower_case=True)

    def __getitem__(self,index):
        # get the sentence from the dataframe
        sentence = self.df.loc[index,'sentence']

        encoded_dict = self.tokenizer.encode_plus(
            sentence,              # sentence to encode
            add_special_tokens = True,         # Add '[CLS]' and '[SEP]'
            max_length = args.max_len,
            pad_to_max_length= True,
            truncation='longest_first',
            return_attention_mask = True,
            return_tensors = 'pt'
        )

        # These are torch tensors already
        input_ids = encoded_dict['input_ids'][0]
        attention_mask = encoded_dict['attention_mask'][0]
        token_type_ids = encoded_dict['token_type_ids'][0]

        #Convert the target to a torch tensor
        target = torch.tensor(self.df.loc[index,'label'])

        sample = (input_ids,attention_mask,token_type_ids,target)
        return sample

    def __len__(self):
        return len(self.df)


def load_test_data(dataset=None):
    test_df = pd.read_csv(f"datasets/{args.dataset}/test.tsv",'\t')
    adv_df = pd.read_csv(f"attack_dataset/{args.task}_{args.attack}_test.tsv",'\t')
    return test_df,adv_df



def getName():

    name = f"{args.dataset}_aug_{args.adv_attack}"
    # add the bachsize information
    name += f"_bs_{args.batch_size}"
    # add the epoch information
    name += f"_ep_{args.epochs}"

    return name



def predict(data_loader,cls):   
    cls.eval()
    y_list = []
    y_hat_list = []
    for batch in tqdm(data_loader):

        batch = tuple(data.cuda() for data in batch)
        inputs_ids, inputs_masks,token_type_ids,inputs_labels = batch
        with torch.no_grad():
            preds = cls(input_ids = inputs_ids, attention_mask=inputs_masks) # 模型预测
        y_list.extend(inputs_labels.detach().cpu().numpy())
        y_hat_list.extend(preds['logits'].detach().cpu().numpy())

    y_list = np.array(y_list)
    y_hat_list = np.array(y_hat_list)
    preds = np.argmax(y_hat_list, axis=1).flatten() # shape = (1, :)
    labels = y_list.flatten()
    acc = np.sum(preds==labels) / len(y_list)
    return acc



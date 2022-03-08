import logging
from torch.utils.data import DataLoader
from torch.nn import functional as F
import pandas as pd
import yaml
import Data


def read_csv_data(file_name):
    dataset = pd.read_csv(file_name, header=0)
    text_list = dataset['text']
    label_list = dataset['label']
    return Data.RawData(text_list, label_list)


def read_yaml_data(file_name):
    file = open(file_name, 'r', encoding='utf-8')
    file_data = file.read()
    file.close()
    data = yaml.load(file_data, Loader=yaml.FullLoader)
    text_list = []
    label_list = []
    for key in data:
        for value in data[key]:
            if value:
                text_list.append(value)
                label_list.append(key)
    return Data.RawData(text_list, label_list)


def convert_text_to_token(tokenizer, text, max_len):
    token = [tokenizer.convert_tokens_to_ids('[CLS]')]
    token += tokenizer.convert_tokens_to_ids(
        word for word in text[:max_len]
    )
    if len(token) < max_len + 1:
        token.extend([0] * (max_len + 1 - len(token)))
    token += [tokenizer.convert_tokens_to_ids('[SEP]')]
    return token


def generate_sample_data(raw_data, tokenizer, label_tab, max_len):
    texts = raw_data.text
    labels = raw_data.label
    id_list = []
    mask_list = []
    label_list = []
    for text, label in zip(texts, labels):
        _id = convert_text_to_token(tokenizer, text, max_len)
        _mask = [float(x > 0) for x in _id]
        if label in label_tab:
            _label = label_tab.index(label)
        else:
            _label = len(label_tab)
            label_tab.append(label)
        id_list.append(_id)
        mask_list.append(_mask)
        label_list.append(_label)
    return Data.Sample(id_list, mask_list, label_list)


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )

    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    # file_handler: write log file
    file_handler = logging.FileHandler(filename, "w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # stream_handler: output log in the terminal
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def generate_batched_data(dataset, batch_size, shuffle=True, drop_last=True, device='cuda:0'):
    dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    )
    for data_dict in dataloader:
        out_dict = {}
        for name, _ in data_dict.items():
            out_dict[name] = data_dict[name].to(device)
        yield out_dict


def calculate_label_loss(pred_label, gold_label):
    return F.cross_entropy(pred_label, gold_label.long(), size_average=False)

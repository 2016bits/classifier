from transformers import BertTokenizer
import torch
import argparse
from Model import BERTModel


def initialize(args):
    label_tab = torch.load(args.info_path)
    tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')
    model = BERTModel(args, len(label_tab)).to(args.device)
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['net'])
    model.eval()
    return tokenizer, model, label_tab


def theme_recognize(args, text, tokenizer, model, label_tab):
    id = [tokenizer.convert_tokens_to_ids('[CLS]')]
    id += tokenizer.convert_tokens_to_ids(
        word for word in text[:args.max_len]
    )
    if len(id) < args.max_len + 1:
        id.extend([0] * (args.max_len + 1 - len(id)))
    id += [tokenizer.convert_tokens_to_ids('[SEP]')]
    mask = [float(x > 0) for x in id]

    scores = model(
        torch.tensor(id).unsqueeze(0).to(args.device),
        torch.tensor(mask).unsqueeze(0).to(args.device)
    )
    prob, index = torch.max(scores, dim=-1)
    label = label_tab[index]
    return label


def main(args):
    tokenizer, model, label_tab = initialize(args)
    while(True):
        text = input("请输入文本：")
        theme = theme_recognize(args, text, tokenizer, model, label_tab)
        print("主题：", theme)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--info_path', type=str, default="./preprocess/theme_info.pt")
    parser.add_argument('--model_path', type=str, default='./model/theme.pth')
    parser.add_argument('--max_len', type=int, default=126)
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--device', type=str, default='cuda:0', choices=['cpu', 'cuda:1'])

    arg = parser.parse_args()
    
    main(arg)
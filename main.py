import pandas as pd
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
import argparse
import torch
import torch.nn as nn
import os
import numpy as np
import Data
import utils
from Model import BERTModel


def test(args, logger, batch_data, model):
    model.eval()
    logger.info('Testing......')
    correct = 0.
    sum = 0.
    for _, batch_dict in enumerate(batch_data):
        scores = model(
            batch_dict['id'].to(args.device),
            batch_dict['mask'].to(args.device)
        )
        prob, index = torch.max(scores, dim=-1)
        for i in range(index.size(0)):
            if index[i] == batch_dict['label'][i]:
                correct += 1
            sum += 1
    acc = correct / sum
    logger.info(
        'correct: {}, sum: {}, acc: {}'.format(correct, sum, acc)
    )
    return acc


def main(args):
    train_path = './' + args.task + args.train_path
    test_path = './' + args.task + args.test_path
    info_path = args.preprocess_data_path + args.task + args.info_path
    log_path = args.log_path + args.task + '.log'
    model_path = args.model_path + args.task + '.pth'

    logger = utils.get_logger(log_path)
    tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')
    # train_data = utils.read_csv_data(args.train_path)
    # test_data = utils.read_csv_data(args.test_path)

    train_data = utils.read_yaml_data(train_path)
    test_data = utils.read_yaml_data(test_path)

    label_tab = []

    # read data
    logger.info("Reading data......")
    # train_list = utils.read_data(dataset[:train_num], tokenizer, label_tab, args.max_len)
    train_list = utils.generate_sample_data(train_data, tokenizer, label_tab, args.max_len)
    train_dataset = Data.BatchedData(train_list)
    train_batch_num = train_dataset.get_batch_num(args.batch_size)
    test_list = utils.generate_sample_data(test_data, tokenizer, label_tab, args.max_len)
    test_dataset = Data.BatchedData(test_list)

    label_tab.append('<PAD>')
    label_tab.append('O')

    # print("label: ", label_tab)
    
    # save data
    torch.save(label_tab, info_path)

    # load model
    logger.info("Loading model......")
    model = BERTModel(args, len(label_tab)).to(args.device)

    logger.info(args)

    # start train
    logger.info("Start training......")
    # optimizer
    logger.info("initial optimizer......")
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if "_bert" in n], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if "_bert" not in n],
        'lr': args.learning_rate, 'weight_decay': 0.01}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.tuning_bert_rate, correct_bias=False)
    
    # load saved model, optimizer and epoch num
    if args.reload and os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info("Reload model and optimizer after training epoch {}".format(start_epoch - 1))
    else:
        start_epoch = 1
        logger.info("New model and optimizer from epoch 0")

    # scheduler
    training_steps = args.epoch_num * len(train_dataset)
    warmup_steps = int(training_steps * args.warm_up)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=training_steps)
    
    best_acc = 0.
    for epoch in range(start_epoch, args.epoch_num + 1):
        model.train()
        model.zero_grad()

        batch_data = utils.generate_batched_data(
            dataset=train_dataset, batch_size=args.batch_size, device=args.device)
        for batch_index, batch_dict in enumerate(batch_data):
            optimizer.zero_grad()
            scores = model(
                batch_dict['id'].to(args.device),
                batch_dict['mask'].to(args.device)
            )
            loss = utils.calculate_label_loss(
                scores,
                batch_dict['label'].view(1, -1).squeeze(0).to(args.device)
            )
            loss.backward()
            optimizer.step()
            scheduler.step()

            if batch_index % 50 == 0:
                logger.info(
                    'Epoch:[{}/{}] Batch:[{}/{}] loss: {}'.format(
                        epoch, args.epoch_num, batch_index, train_batch_num,
                        round(loss.item(), 4)
                    )
                )
        
        batch_data = utils.generate_batched_data(
            dataset=test_dataset, batch_size=args.batch_size, device=args.device)
        acc = test(args, logger, batch_data, model)
        if acc > best_acc:
            best_acc = acc
            logger.info('Model saved after epoch {}, acc: {}'.format(epoch, acc))
            state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(state, model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='theme', choices=['intent', 'sentiment'])
    # theme recognize
    # # csv
    # parser.add_argument('--train_path', type=str, default="_train.csv")
    # parser.add_argument('--test_path', type=str, default='_test.csv')
    # yaml
    parser.add_argument('--train_path', type=str, default="_train.yaml")
    parser.add_argument('--test_path', type=str, default='_test.yaml')

    parser.add_argument('--preprocess_data_path', type=str, default="./preprocess/")
    parser.add_argument('--info_path', type=str, default="_info.pt")
    parser.add_argument('--log_path', type=str, default="./logger/logger_")
    parser.add_argument('--max_len', type=int, default=126)
    parser.add_argument('--hidden_size', type=int, default=768)

    parser.add_argument('--reload', type=bool, default=False)
    parser.add_argument('--model_path', type=str, default='./model/')

    parser.add_argument('--device', type=str, default='cuda:0', choices=['cpu', 'cuda:1'])
    parser.add_argument('--epoch_num', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--tuning_bert_rate', type=float, default=1e-5)
    parser.add_argument('--warm_up', type=float, default=0.1)

    arg = parser.parse_args()

    main(arg)

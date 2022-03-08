from transformers import BertModel, BertConfig
import torch.nn as nn


class BERTModel(nn.Module):
    def __init__(self, args, num):
        self.hidden_size = args.hidden_size
        self.device = args.device

        super(BERTModel, self).__init__()

        self.model_config = BertConfig.from_pretrained('./bert-base-chinese')
        self.model_config.output_hidden_states = True
        self.model_config.output_attentions = True
        self._bert = BertModel.from_pretrained('./bert-base-chinese', config=self.model_config).to(self.device)

        self._classifier = nn.Linear(self.hidden_size, num).to(self.device)

    def forward(self, text_id, text_mask):
        hidden_states = self._bert(text_id, attention_mask=text_mask)[0]
        text_class_scores = self._classifier(hidden_states[:, 0, :])
        return text_class_scores

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class Datasets(Dataset):
    def __init__(self, texts, targets, bert_dir, device):
        self.tokenizer = AutoTokenizer.from_pretrained(bert_dir)  # 分词模型
        assert len(texts) == len(targets)
        self.texts = texts
        self.targets = targets
        self.device = device

    def __getitem__(self, idx):
        text = self.texts[idx]
        target = self.targets[idx]
        return (text, target)

    def __len__(self):
        return len(self.targets)

    def collate_fn(self, batch):
        batch_texts = [text for (text, target) in batch]
        batch_targets = [target for (text, target) in batch]

        batch_token_ripe = self.tokenizer.batch_encode_plus(
            batch_texts,
            padding=True,
            return_offsets_mapping=True,
        )  # bert分词 padding到该batch的最大长度

        return (
            torch.LongTensor(batch_token_ripe["input_ids"]).to(self.device),
            torch.ByteTensor(batch_token_ripe["attention_mask"]).to(self.device),
            torch.FloatTensor(batch_targets).to(self.device),
            batch_texts,
        )

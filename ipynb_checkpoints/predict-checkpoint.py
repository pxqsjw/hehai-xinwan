# -- coding: utf-8 --
import torch
from transformers import AutoTokenizer

from action_opt import configs
from model import BertModel
from pretreatment import text_pretreatment

TOKENIZER = AutoTokenizer.from_pretrained(configs.BERT_DIR)  # bert分词器
MODEL = BertModel(configs.BERT_DIR, outputs_num=len(configs.LABEL_SETS))  # 模型
MODEL.load_state_dict(torch.load(configs.BERT_MODEL_PATH, map_location=torch.device("cpu")))  # 加载模型参数
MODEL.to(configs.DEVICE)
MODEL.eval()  # 验证模式
print("previous model loading...")


def to_inputs(batch_texts):
    batch_token_ripe = TOKENIZER.batch_encode_plus(
        batch_texts,
        padding=True,
        return_offsets_mapping=True,
    )  # bert分词 padding到该batch的最大长度
    return (
        torch.LongTensor(batch_token_ripe["input_ids"]).to(configs.DEVICE),
        torch.ByteTensor(batch_token_ripe["attention_mask"]).to(configs.DEVICE),
    )


def predict(batch_text):
    batch_texts = [text_pretreatment(text) for text in batch_text]  # 预处理
    batch_inputs, batch_mask = to_inputs(batch_texts)  # to token
    with torch.no_grad():
        outputs = MODEL(batch_inputs, attention_mask=batch_mask)  # 预测
    preds = (torch.sigmoid(outputs) > 0.5).float().cpu().numpy().astype(int).tolist()
    results = [
        " / ".join([configs.LABEL_SETS[idx] for idx, item in enumerate(result) if item == 1]) for result in preds
    ]
    return results, preds


if __name__ == "__main__":
    text = """#昨晚的微博评论太好哭了##1分钟看郑州暴雨救援感人瞬间# 应该让我们的孩子，在一次次社会课堂中，感受那些美好的中国精神，倾听那些颇具力量的声音，敬仰那些勇敢奔赴的身影，悲悯那些美好生命的逝去。这个社会、国家、世界，终究属于他们，他们需要用这些最好的东西武装成长，眼睛里流露的应是想缔造更好未来的星星✨希望我们的家长加以引导，抓住一些教育时机，也是增加亲子沟通的机会，暑假生活有见有闻，有行有思，潜移默化地影响咱们的孩子（来自班主任的日常唠叨） 宜昌 """
    outputs = predict([text])[0]
    print("预测标签为:", outputs)

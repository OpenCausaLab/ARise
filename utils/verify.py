import logging
import re
from collections import Counter


def exact_match(pred_answer, true_answer):
    return true_answer.lower() in pred_answer.lower()


def normalize_text(s):
    """规范化文本"""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
        pattern = "[" + "".join("\\" + char for char in exclude) + "]"
        return re.sub(pattern, "", text)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    """获取标准化后的tokens"""
    if not s:
        return []
    return normalize_text(s).split()


def F1_score_compute(gt_facts, retrieval_facts):
    """
    计算检索结果与真实事实之间的F1分数
    Args:
        gt_facts (str): 真实事实字符串
        retrieval_facts (str): 检索到的事实字符串
    Returns:
        float: F1分数
    """
    try:
        # 获取tokens
        pred_tokens = get_tokens(retrieval_facts)
        gold_tokens = get_tokens(gt_facts)

        if not pred_tokens or not gold_tokens:
            return 0.0

        # 计算共同tokens
        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            return 0.0

        # 计算精确率和召回率
        precision = 1.0 * num_same / len(pred_tokens)
        recall = 1.0 * num_same / len(gold_tokens)

        # 计算F1
        f1 = (2 * precision * recall) / (precision + recall)

        return f1

    except Exception as e:
        logging.error(f"Error in F1 score calculation: {str(e)}")
        return 0.0

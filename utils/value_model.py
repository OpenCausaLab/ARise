import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Global variables for model caching
VALUE_MODEL_DIR = "../hfmodels/Qwen/Qwen2.5-72B-Instruct"
global_value_model = None
global_tokenizer = None


def initialize_value_model():
    """Initialize the value model and tokenizer."""
    global global_value_model, global_tokenizer

    if global_value_model is not None and global_tokenizer is not None:
        return True  # Model already initialized

    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(VALUE_MODEL_DIR)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            VALUE_MODEL_DIR,
            torch_dtype=torch.bfloat16,
            # device_map="auto",  # Automatically choose best device
        )

        # Move model to GPU if available
        if torch.cuda.is_available():
            model = model.cuda()

        global_value_model = model
        global_tokenizer = tokenizer

        print("Value model initialized successfully")
        return True

    except Exception as e:
        print(f"Error initializing value model: {str(e)}")
        return False


def cleanup_value_model():
    """Cleanup model resources."""
    global global_value_model, global_tokenizer

    if global_value_model is not None:
        del global_value_model
        global_value_model = None

    if global_tokenizer is not None:
        del global_tokenizer
        global_tokenizer = None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Value model resources cleaned up")


def get_token_probabilities(text, idx, inputs=None):
    """
    Calculate log probabilities for tokens from idx onwards.
    Each probability p(d_t|d_<t) is conditioned only on previous tokens.

    Args:
        text (str): Input text sequence d
        idx (int): Starting index for probability calculation
        inputs (dict, optional): Pre-tokenized inputs, if None will tokenize text

    Returns:
        list: List of log probabilities [log p(d_t|d_<t)] for t >= idx
    """
    global global_value_model, global_tokenizer

    if (
        global_value_model is None or global_tokenizer is None
    ) and not initialize_value_model():
        return []

    try:
        # Use pre-tokenized inputs if provided, otherwise tokenize text
        if inputs is None:
            inputs = global_tokenizer(
                text, padding=True, truncation=True, max_length=512, return_tensors="pt"
            )

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Move to GPU if available
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()

        log_probs = []
        with torch.no_grad():
            # Get model outputs for the entire sequence
            outputs = global_value_model(
                input_ids=input_ids, attention_mask=attention_mask
            )
            logits = outputs.logits[0]  # Remove batch dimension

            # Calculate log probabilities for each position from idx
            for pos in range(
                idx - 1, input_ids.shape[1] - 1
            ):  # -1 because we predict next token
                # Get log probabilities for next token
                next_token_logits = logits[pos]
                log_probs_t = torch.log_softmax(next_token_logits, dim=-1)

                # Get log probability of the actual next token
                next_token_id = input_ids[0, pos + 1]
                log_prob = log_probs_t[next_token_id].item()
                log_probs.append(log_prob)

        return log_probs

    except Exception as e:
        print(f"Error calculating token probabilities: {str(e)}")
        return []


def get_query_token_probabilities(context, query):
    """
    获取query部分的token概率

    Args:
        context (str): 前文上下文
        query (str): 要计算概率的查询文本

    Returns:
        list: query部分token的log概率列表
    """
    global global_value_model, global_tokenizer

    if (
        global_value_model is None or global_tokenizer is None
    ) and not initialize_value_model():
        return []

    try:
        # 先对整个文本做tokenization
        full_text = context + query
        inputs = global_tokenizer(
            full_text, padding=True, truncation=True, return_tensors="pt"
        )

        # 单独对前文做tokenization，找到query的起始位置
        context_tokens = global_tokenizer(
            context, padding=False, truncation=False, return_tensors="pt"
        )
        query_start_idx = context_tokens["input_ids"].shape[1]

        # 获取query部分的概率，传入已tokenized的inputs
        return get_token_probabilities(full_text, query_start_idx, inputs)

    except Exception as e:
        print(f"Error in get_query_token_probabilities: {str(e)}")
        return []


# Initialize the model when module is imported
initialize_value_model()

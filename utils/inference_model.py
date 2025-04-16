import torch
import transformers

# Global variables for model caching
INFERENCE_LOCAL = True
INFERENCE_MODEL_DIR = "../hfmodels/Qwen/Qwen2.5-72B-Instruct"
global_pipline = None


def initialize_model():
    global global_pipline
    if global_pipline is not None:
        return True
    try:
        pipeline = transformers.pipeline(
            "text-generation",
            model=INFERENCE_MODEL_DIR,
            torch_dtype=torch.bfloat16,
            device_map="balanced_low_0",
        )
        global_pipline = pipeline
        print("Model initialized successfully")
        return True
    except Exception as e:
        print(f"Error during model initialization: {str(e)}")
        return False


def cleanup_model():
    """Cleanup model resources."""
    global global_pipline
    if global_pipline is None:
        del global_pipline
        global_pipline = None
    torch.cuda.empty_cache()
    global_pipline = None
    print("Model resources cleaned up")


def get_response(
    prompt,
    temperature=0.7,
    max_tokens=2048,
    seed=170,
    max_length=2048,
    truncation=True,
    do_sample=True,
    max_new_tokens=1024,
    num_return_sequences=1,
):
    """Get response from the model with retries."""
    response = []
    cnt = 2
    while not response and cnt:
        response = local_inference_model(
            prompt,
            max_length=max_length,
            truncation=truncation,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
        )
        cnt -= 1
    if not response:
        print("Failed to obtain response")
        return []
    return response


def local_inference_model(
    query,
    max_length=2048,
    truncation=True,
    do_sample=False,
    max_new_tokens=1024,
    temperature=0.7,
    num_return_sequences=1,
):
    """Local inference using cached model."""
    global global_pipline
    assert global_pipline is not None, "Model not initialized"

    return get_local_response_llama(
        query,
        global_pipline,
        max_length=max_length,
        truncation=truncation,
        do_sample=do_sample,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        num_return_sequences=num_return_sequences,
    )


def get_local_response_llama(
    query,
    pipeline,
    max_length=2048,
    truncation=True,
    max_new_tokens=1024,
    temperature=0.7,
    do_sample=False,
    num_return_sequences=1,
):
    """
    Generate response using Llama model with flexible configuration.

    Args:
        query (str): Input query or prompt
        pipeline: HuggingFace pipeline
        max_length (int): Maximum sequence length
        truncation (bool): Whether to truncate long sequences
        max_new_tokens (int): Maximum new tokens to generate
        temperature (float): Sampling temperature
        do_sample (bool): Whether to use sampling
        num_return_sequences (int): Number of different sequences to generate

    Returns:
        List[str]: Generated responses
    """
    try:
        # Prepare input for model
        message = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": query},
        ]

        outputs = pipeline(
            message,
            max_length=max_length,
            truncation=truncation,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
        )

        # Process outputs
        responses = []
        for output in outputs:
            response = output["generated_text"][-1]
            if query in response:
                response = response.replace(query, "").strip()
            responses.append(response)

        return responses

    except Exception as e:
        print(f"Error generating response: {e}")
        return []


# Initialize the model
initialize_model()

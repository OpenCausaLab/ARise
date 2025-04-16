import logging

from utils.inference_model import get_response
from utils.prompts import tot_value_prompt
from utils.wrap import (
    tot_answer_wrap,
    tot_wrap,
    zero_shot_proposal_wrap,
)


def get_answer(prompt, task, n=1):
    all_responses = []
    max_retries = 3
    attempts = 0
    response = []

    while not response and attempts < max_retries:
        try:
            response = get_response(
                prompt,
                temperature=task.temperature,
                max_tokens=task.max_tokens,
                seed=task.seed,
                max_length=task.max_length,
                truncation=task.truncation,
                do_sample=task.do_sample,
                max_new_tokens=task.max_new_tokens,
                num_return_sequences=n,
            )
        except Exception as e:
            logging.error(
                f"Error getting model response (attempt {attempts + 1}): {str(e)}"
            )
        attempts += 1

    if not response:
        logging.error("Failed to get reflection after all retries")
        return ""

    # Process response
    try:
        if isinstance(response, list) and response and isinstance(response[0], dict):
            for element in response:
                proposal = element.get("content", "")
                all_responses.append(proposal)
        elif isinstance(response, list) and response and isinstance(response[0], str):
            for element in response:
                proposal = element
                all_responses.append(proposal)
        else:
            logging.error(f"Unexpected response format: {response}")
            return ""

        return all_responses

    except Exception as e:
        logging.error(f"Error processing model response: {str(e)}")
        return ""


def baseline_task(task, run_mode):
    if run_mode == "zero-shot":
        prompt, knowledge = zero_shot_proposal_wrap(
            task.data["question"], task.data_path, task.data_idx
        )

    else:
        raise ValueError(f"Invalid run mode: {run_mode}")

    answer = get_answer(prompt, task, 1)
    answer = answer[0] if answer else "N/A"

    if not answer:
        answer = "N/A"

    data = {"question": task.data["question"], "answer": answer, "facts": knowledge}

    return data


def ToT_task(task):
    init_state = ""
    question = task.data["question"]
    first_layer = {}
    second_layer = {}
    third_layer = {}
    prompt = tot_wrap(question, init_state, task.data_path, task.data_idx)
    responses = get_answer(prompt, task, 10)
    for response in responses:
        if "First:" and "Second:" and "Third:" in response:
            step = response.split("Second:")[0].strip()
            score_prompt = tot_value_prompt.format(question=question, state=step)
            score = get_answer(score_prompt, task, 1)
            try:
                score = float(score[0])
            except ValueError:
                score = 0.0
            first_layer[step] = score
    # 根据score对first_layer进行排序并保留前三名
    first_layer = {
        k: v
        for k, v in sorted(first_layer.items(), key=lambda item: item[1], reverse=True)
    }
    first_layer = list(first_layer.keys())[:3]

    for state in first_layer:
        prompt = tot_wrap(question, state, task.data_path, task.data_idx)
        responses = get_answer(prompt, task, 4)
        for response in responses:
            if "Second:" and "Third:" in response:
                step = response.split("Third:")[0].strip()
                score_prompt = tot_value_prompt.format(
                    question=question, state=state + step
                )
                score = get_answer(score_prompt, task, 1)
                try:
                    score = float(score[0])
                except ValueError:
                    score = 0.0
                second_layer[state + step] = score

    second_layer = {
        k: v
        for k, v in sorted(second_layer.items(), key=lambda item: item[1], reverse=True)
    }
    second_layer = list(second_layer.keys())[:3]

    for state in second_layer:
        prompt = tot_wrap(question, state, task.data_path, task.data_idx)
        responses = get_answer(prompt, task, 4)
        for response in responses:
            if "Third:" in response:
                step = response.split("Third:")[1].strip()
                step = "Third: " + step
                score_prompt = tot_value_prompt.format(
                    question=question, state=state + step
                )
                score = get_answer(score_prompt, task, 1)
                try:
                    score = float(score[0])
                except ValueError:
                    score = 0.0
                third_layer[state + step] = score

    third_layer = {
        k: v
        for k, v in sorted(third_layer.items(), key=lambda item: item[1], reverse=True)
    }
    third_layer = list(third_layer.keys())[0]

    prompt, knowledge = tot_answer_wrap(
        question, third_layer, task.data_path, task.data_idx
    )
    final_answer = get_answer(prompt, task, 1)
    final_answer = final_answer[0] if final_answer else "N/A"
    if not final_answer:
        final_answer = "N/A"

    data = {
        "question": task.data["question"],
        "answer": final_answer,
        "facts": knowledge if knowledge else "N/A",
    }

    return data

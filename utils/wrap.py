from utils.prompts import (
    answer_prompt,
    decompose_prompt,
    final_answer_prompt,
    tot_answer_prompt,
    tot_thought_prompt,
    zero_shot_proposal_prompt,
)
from utils.rag import retrieve


def decompose_wrap(state, data_path, data_idx) -> str:
    print("\n", "=" * 30, "decompose sub-queries", "=" * 30, "\n")
    print(f"Current question:\n{state}")

    external_knowledge = retrieve(state, data_path, data_idx)
    if not external_knowledge:
        external_knowledge = "No directly relevant facts found."
    # Format the complete prompt using the template with placeholders
    prompt = decompose_prompt.format(state=state, knowledge=external_knowledge)

    return prompt


def answer_wrap(question, data_path, data_idx) -> str:
    print("\n", "=" * 30, "answer question", "=" * 30, "\n")
    print(f"Current question:\n{question}")

    external_knowledge = retrieve(question, data_path, data_idx)
    if not external_knowledge:
        external_knowledge = "No directly relevant facts found."

    # Format the complete prompt using the template with placeholders
    prompt = answer_prompt.format(
        question=question, external_knowledge=external_knowledge
    )

    return prompt, external_knowledge


def final_answer_wrap(state, data_path, data_idx) -> str:
    print("\n", "=" * 30, "answer final question", "=" * 30, "\n")
    print(f"Current question:\n{state}")

    external_knowledge = retrieve(state, data_path, data_idx)
    if not external_knowledge:
        external_knowledge = "No directly relevant facts found."

    # Format the complete prompt using the template with placeholders
    prompt = final_answer_prompt.format(state=state)

    return prompt, external_knowledge


def zero_shot_proposal_wrap(question, data_path, data_idx) -> str:
    external_knowledge = retrieve(question, data_path, data_idx)
    if not external_knowledge:
        external_knowledge = "No directly relevant facts found."
    prompt = zero_shot_proposal_prompt.format(
        question=question, external_knowledge=external_knowledge
    )
    return prompt, external_knowledge


def tot_wrap(question, state, data_path, data_idx) -> str:
    external_knowledge = retrieve(f"{question} {state}", data_path, data_idx)
    if not external_knowledge:
        external_knowledge = "No directly relevant facts found."
    prompt = tot_thought_prompt.format(
        question=question, state=state, external_knowledge=external_knowledge
    )
    return prompt


def tot_answer_wrap(question, state, data_path, data_idx) -> str:
    external_knowledge = retrieve(f"{question} {state}", data_path, data_idx)
    if not external_knowledge:
        external_knowledge = "No directly relevant facts found."
    prompt = tot_answer_prompt.format(
        question=question, state=state, external_knowledge=external_knowledge
    )
    return prompt, external_knowledge

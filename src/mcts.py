import math
import random
import time

import numpy

from src.node import TreeNode


def MCTS_search(mcts_task):
    """
    Performs Monte Carlo Tree Search within specified time or iteration limits.
    Returns the root node, solution node (if found), and search duration/iterations.
    """
    # Validate and set search limits
    mcts_task.set_limit()

    root_node = TreeNode(
        query=mcts_task.data["question"], num_children=mcts_task.multihops
    )
    search_start_time = time.time()

    if mcts_task.limit_type == "time":
        search_end_time = search_start_time + mcts_task.time_limit / 1000
        while time.time() < search_end_time:
            print(
                f"<begin a new search round, elapsed time: {time.time() - search_start_time:.2f}s>\n"
            )
            root_node = execute_round(root_node, mcts_task)
        search_metric = time.time() - search_start_time

    else:
        for iteration_count in range(mcts_task.iteration_limit):
            print(
                f"<Begin search round {iteration_count + 1}/{mcts_task.iteration_limit}>\n"
            )
            root_node = execute_round(root_node, mcts_task)
        search_metric = mcts_task.iteration_limit

    return root_node, search_metric


def execute_round(root_node, mcts_task):
    # Execute a selection-expansion-simulation-backpropagation round
    print("-" * 30, "phase selection", "-" * 30, "\n")
    selected_node = select_node(root_node, mcts_task)
    print(f"Selected node: {selected_node.state}, depth: {selected_node.depth}\n")

    print("-" * 30, "phase expansion", "-" * 30, "\n")
    if selected_node.is_terminal:
        print("This is a terminal node, no further expansion required.\n")
    else:
        selected_node = expand_node(selected_node, mcts_task)
        print(
            f"Complete expansion!, expanded node count: {len(selected_node.children)}\n"
        )

    print("-" * 30, "phase simulation", "-" * 30, "\n")
    if selected_node.is_terminal:
        print("This is a terminal node, no further simulation required.\n")
    else:
        rollout_node = get_best_child(selected_node, mcts_task)
        best_value = greedy_policy(rollout_node, mcts_task)
        # Update value with exponential moving average
        rollout_node.value = (
            rollout_node.value * (1 - mcts_task.alpha) + best_value * mcts_task.alpha
        )
        rollout_node.visit_count += 1

    print("-" * 30, "phase backpropagation", "-" * 30, "\n")
    back_propagate(selected_node)

    return root_node


def select_node(current_node, mcts_task):
    while current_node.is_fully_expanded:
        current_node = get_best_child(current_node, mcts_task)
    return current_node


def get_best_child(parent_node, mcts_task):
    best_value = mcts_task.low_value
    best_child_nodes = []
    for child_node in parent_node.children.values():
        # UCB1 formula for node selection
        if child_node.visit_count > 0:
            exploitation_term = child_node.value
            exploration_term = mcts_task.exploration_constant * math.sqrt(
                2 * math.log(parent_node.visit_count) / child_node.visit_count
            )
            ucb_value = exploitation_term + exploration_term
        else:
            ucb_value = child_node.value + 1.0  # 确保未访问的节点会被选中

        if ucb_value > best_value:
            best_value = ucb_value
            best_child_nodes = [child_node]
        elif ucb_value == best_value:
            best_child_nodes.append(child_node)
    return random.choice(best_child_nodes)


def expand_node(current_node, mcts_task):
    possible_sub_nodes = get_sub_nodes(
        current_node.state, current_node.num_children, mcts_task
    )
    if not possible_sub_nodes:
        current_node.is_terminal = True
        return current_node

    for sub_node in possible_sub_nodes:
        if list(sub_node.keys())[0] not in list(current_node.children.keys()):
            sub_query = list(sub_node.keys())[0]
            sub_answer = sub_node[sub_query]
            current_node.append_children(sub_query, sub_answer)
            child_node = current_node.children[sub_query]
            # knowledge = sub_node["external_knowledge"]
            child_node.update_value(
                mcts_task.get_node_value(
                    get_all_sub_queries(child_node), get_all_sub_answers(child_node)
                )
            )

    current_node.is_fully_expanded = True
    return current_node


def get_all_sub_queries(node):
    sub_queries = []
    while node.parent is not None:
        sub_queries.append(node.query)
        node = node.parent
    return "\n".join(reversed(sub_queries))


def get_all_sub_answers(node):
    sub_ans = []
    while node.parent is not None:
        sub_ans.append(node.answer)
        node = node.parent
    return "\n".join(reversed(sub_ans))


def get_sub_nodes(state, num_children, mcts_task):
    proposed_sub_nodes = []
    remaining_attempts = 3
    while not proposed_sub_nodes and remaining_attempts > 0:
        proposed_sub_nodes = mcts_task.sub_queries_to_nodes(state, num_children)
        remaining_attempts -= 1
    return proposed_sub_nodes


def greedy_policy(current_node, mcts_task):
    max_value = mcts_task.low_value

    if current_node.is_terminal:
        print("This step has solved the problem and does not require simulation.\n")
        return current_node.value

    cur_state = current_node.state
    num_children = current_node.num_children
    roll_steps = mcts_task.total_depth - current_node.depth - 1
    cur_query = current_node.query
    cur_ans = current_node.answer

    for _ in range(roll_steps):
        possible_sub_nodes = get_sub_nodes(cur_state, num_children, mcts_task)
        if not possible_sub_nodes:
            current_node.is_terminal = True
            break

        candidate_values = []
        for node in possible_sub_nodes:
            query = list(node.keys())[0]
            answer = node[query]
            # knowledge = node["external_knowledge"]
            value = mcts_task.get_node_value(
                cur_query + "\n" + query, cur_ans + "\n" + answer
            )
            candidate_values.append(value)

        best_candidate_idx = numpy.argmax(candidate_values)
        sub_node = possible_sub_nodes[best_candidate_idx]
        sub_query = list(sub_node.keys())[0]
        sub_answer = sub_node[sub_query]

        cur_state += f"Intermediate answer: {sub_answer}\n"
        cur_query = cur_query + "\n" + sub_query
        cur_ans = cur_ans = "\n" + sub_answer
        num_children -= 1

        cur_value = candidate_values[best_candidate_idx]
        max_value = max(max_value, cur_value)

    return max_value


def back_propagate(selected_node):
    current_node = selected_node
    while current_node is not None:
        current_node.visit_count += 1
        if current_node.is_fully_expanded:
            child_weighted_values = [
                child.value * child.visit_count
                for child in current_node.children.values()
            ]
            total_child_visits = sum(
                child.visit_count for child in current_node.children.values()
            )
            if total_child_visits > 0:
                current_node.value = sum(child_weighted_values) / total_child_visits
        current_node = current_node.parent

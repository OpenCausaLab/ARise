import logging
import random

import numpy as np

from src.base import get_answer
from src.mcts import MCTS_search
from utils.inference_model import get_response
from utils.prompts import star_value_prompt
from utils.value_function import risk_value, similarity_value
from utils.wrap import answer_wrap, decompose_wrap, final_answer_wrap


class MCTSTask:
    def __init__(
        self,
        time_limit=None,  # Time limit in milliseconds
        iteration_limit=None,  # Maximum number of iterations
        exploration_constant=1.0,  # UCT exploration constant
        data=None,  # test data
        data_path=None,
        data_idx=None,
        alpha=0.1,  # Value update rate
        ans_weight=0.75,  # knwledge weight
        multihops=6,  # sub query
        total_depth=5,  # Total depth of the tree
        temperature=0.7,  # Sampling temperature
        max_tokens=2048,  # Max tokens for generation
        seed=170,  # Random seed
        max_length=2048,  # Max sequence length
        truncation=True,  # Whether to truncate
        do_sample=True,  # Whether to sample
        max_new_tokens=1024,  # Max new tokens to generate
        low=0,  # Minimum value
        high=1,  # Maximum value
        run_mode="MCTS",
        value_mode="risk",
        value_model="Qwen2.5-14B-Instruct",
    ):
        # Task parameters
        self.run_mode = run_mode
        self.value_mode = value_mode
        self.value_model = value_model

        self.time_limit = time_limit
        self.iteration_limit = iteration_limit
        self.exploration_constant = exploration_constant

        # Model parameters
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.max_length = max_length
        self.truncation = truncation
        self.do_sample = do_sample
        self.max_new_tokens = max_new_tokens

        # State
        self.data = data
        self.data_path = data_path
        self.data_idx = data_idx
        self.node_count = 0
        self.limit_type = time_limit
        self.root_node = None
        self.leaf_nodes = []

        # Value range
        self.low_value = low
        self.high_value = high
        self.alpha = alpha
        self.ans_weight = ans_weight
        self.multihops = multihops
        self.total_depth = total_depth

    def set_limit(self):
        """Set and validate the search limit type (time or iterations).

        Raises:
            ValueError: If both time and iteration limits are set, or if neither is set,
                      or if iteration limit is less than 1.
        """
        if self.time_limit is not None and self.iteration_limit is not None:
            raise ValueError("Cannot have both a time limit and an iteration limit")

        if self.time_limit is None and self.iteration_limit is None:
            raise ValueError("Must have either a time limit or an iteration limit")

        if self.time_limit is not None:
            self.limit_type = "time"
        else:
            if self.iteration_limit < 1:
                raise ValueError("Iteration limit must be greater than one")
            self.limit_type = "iterations"

    def sub_queries_to_nodes(self, state, num_children):
        """
        Generate sub-queries from the current node.
        """
        # Generate decompose prompt
        try:
            decompose_prompt = decompose_wrap(state, self.data_path, self.data_idx)
        except Exception as e:
            logging.error(f"Error generating proposal prompt: {str(e)}")
            return ""

        # Get model response with retries
        max_retries = 3
        attempts = 0
        response = []

        while not response and attempts < max_retries:
            try:
                response = get_response(
                    decompose_prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    seed=self.seed,
                    max_length=self.max_length,
                    truncation=self.truncation,
                    do_sample=self.do_sample,
                    max_new_tokens=self.max_new_tokens,
                    num_return_sequences=num_children,
                )
            except Exception as e:
                logging.error(
                    f"Error getting model response (attempt {attempts + 1}): {str(e)}"
                )
            attempts += 1

        if not response:
            logging.error("Failed to get next step after all retries")
            return ""

        # Process response
        proposed_sub_queries = []
        proposals = []

        try:
            if (
                isinstance(response, list)
                and response
                and isinstance(response[0], dict)
            ):
                for element in response:
                    proposal = element.get("content", "")
                    proposals.append(proposal)

            elif (
                isinstance(response, list) and response and isinstance(response[0], str)
            ):
                for element in response:
                    proposal = element
                    proposals.append(proposal)

            else:
                logging.error("Invalid output format of sub-queries, not a list")
                return ""

            for proposal in proposals:
                if "Sub-question:" not in proposal:
                    logging.error(
                        "Invalid output format - missing 'Sub-question:' marker"
                    )
                    continue

                sub_query = proposal.split("Sub-question:")[1].strip()

                if (
                    len(sub_query) < 10
                ):  # Arbitrary minimum length for a meaningful step
                    logging.error("Generated sub-query too short to be meaningful")
                    continue

                if sub_query in state:
                    logging.error(
                        "Generated sub-query is a duplicate of original query"
                    )
                    continue

                proposed_sub_queries.append(sub_query)

        except Exception as e:
            logging.error(f"Error processing model response: {str(e)}")
            return ""

        revised_sub_queries = []

        for sub_query in proposed_sub_queries:
            tmp_dict = {}
            sub_query_prompt, external_knowledge = answer_wrap(
                sub_query, self.data_path, self.data_idx
            )

            max_retries = 3
            attempts = 0
            response = []

            while not response and attempts < max_retries:
                try:
                    response = get_response(
                        sub_query_prompt,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        seed=self.seed,
                        max_length=self.max_length,
                        truncation=self.truncation,
                        do_sample=self.do_sample,
                        max_new_tokens=self.max_new_tokens,
                        num_return_sequences=1,
                    )
                except Exception as e:
                    logging.error(
                        f"Error getting model response (attempt {attempts + 1}): {str(e)}"
                    )
                attempts += 1

            if not response:
                logging.error("Failed to get next step after all retries")
                continue

            # Process response
            try:
                if (
                    isinstance(response, list)
                    and response
                    and isinstance(response[0], dict)
                ):
                    proposal = response[0].get("content", "")

                elif (
                    isinstance(response, list)
                    and response
                    and isinstance(response[0], str)
                ):
                    proposal = response[0]
                else:
                    logging.error("Invalid output format of sub-queries, not a list")
                    continue

                # Validate step quality
                if len(proposal) < 6:  # Arbitrary minimum length for a meaningful step
                    logging.error("Generated step too short to be meaningful")
                    continue

                if "No directly relevant facts found" in proposal:
                    continue

                if proposal in state:
                    continue

                if "No directly relevant facts found" in proposal:
                    continue

                tmp_dict[sub_query] = proposal
                tmp_dict["external_knowledge"] = external_knowledge
                revised_sub_queries.append(tmp_dict)

            except Exception as e:
                logging.error(f"Error processing model response: {str(e)}")
                continue

        return revised_sub_queries

    def get_node_value(self, query, answer):
        """
        Calculate weighted combination of:
        1. TF-IDF similarity between query-answer pair and original question
        2. TF-IDF similarity between knowledge and query to assess knowledge reliability

        Returns:
            float: Combined similarity score
        """
        if self.value_mode == "sim":
            return similarity_value(
                self.data["question"], query, answer, self.ans_weight
            )

        elif self.value_mode == "risk":
            return risk_value(self.data["question"], query, answer, self.ans_weight)

        elif self.value_mode == "random":
            return random.uniform(self.low_value, self.high_value)

        elif self.value_mode == "star":
            prompt = star_value_prompt.format(
                question=self.data["question"], sub=query, ans=answer
            )
            response = get_answer(prompt, self, 1)
            response = response[0] if response else ""
            try:
                score = float(response)
            except ValueError:
                score = 0.0
            return score

        else:
            raise ValueError(f"Invalid value mode: {self.value_mode}")

    def run(self):
        """
        Run MCTS search.

        Returns:
            TreeNode: Root node of the search tree
        """
        try:
            root_node, search_metric = MCTS_search(self)
            self.root_node = root_node  # Store for class-level access if needed
            print(f"Search completed with {search_metric} iterations")
            return root_node
        except Exception as e:
            logging.error(f"Error during MCTS search: {str(e)}")
            raise

    def traverse_tree(self, node):
        """
        Traverse the tree and save results to a JSON file.

        Args:
            node: Current node to traverse. If None, starts from root_node
            output_file: Path to output file
        """
        if node is None:
            node = self.root_node

        if node is None:
            return

        try:
            if node.children:
                for child in node.children:
                    self.traverse_tree(node.children[child])

            path_q = []
            cur_node = node
            while cur_node.parent is not None:
                path_q.append(cur_node.query)
                cur_node = cur_node.parent

            path_q.append(cur_node.query)

            final_answer_prompt, knowledge = final_answer_wrap(
                node.state, self.data_path, self.data_idx
            )

            max_retries = 3
            attempts = 0
            response = []

            while not response and attempts < max_retries:
                try:
                    response = get_response(
                        final_answer_prompt,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        seed=self.seed,
                        max_length=self.max_length,
                        truncation=self.truncation,
                        do_sample=self.do_sample,
                        max_new_tokens=self.max_new_tokens,
                        num_return_sequences=1,
                    )
                except Exception as e:
                    logging.error(
                        f"Error getting model response (attempt {attempts + 1}): {str(e)}"
                    )
                attempts += 1

            if not response:
                logging.error("Failed to get reflection after all retries")
                return

            # Process response
            try:
                if (
                    isinstance(response, list)
                    and response
                    and isinstance(response[0], dict)
                ):
                    proposal = response[0].get("content", "")
                elif (
                    isinstance(response, list)
                    and response
                    and isinstance(response[0], str)
                ):
                    proposal = response[0]
                else:
                    logging.error("Invalid output format of reflection, not a list")
                    return

                final_answer = proposal

            except Exception as e:
                logging.error(f"Error processing model response: {str(e)}")
                return

            self.leaf_nodes.append(
                {
                    "original query": self.data["question"],
                    "last_query": node.state,
                    "answer": final_answer,
                    "facts": knowledge,
                    "path": path_q,
                    "value": node.value,
                }
            )

        except Exception as e:
            logging.error(f"Error while traversing tree: {str(e)}")
            raise

        return self.leaf_nodes

    def get_best_path(self):
        """
        Get the leaf node with the highest value.

        Returns:
            TreeNode: Leaf node with the highest value
        """
        current_node = self.root_node
        path_q = []
        while not current_node.is_terminal:
            if current_node.children:
                candidate_values = []
                for child in current_node.children.values():
                    candidate_values.append(child.value)
                best_candidate_idx = np.argmax(candidate_values)
                current_node = current_node.children[
                    list(current_node.children.keys())[best_candidate_idx]
                ]
            else:
                current_node.value = 0
                current_node = current_node.parent

        path_node = current_node
        while path_node.parent is not None:
            path_q.append(path_node.query)
            path_node = path_node.parent

        path_q.append(path_node.query)

        final_answer_prompt, knowledge = final_answer_wrap(
            current_node.state, self.data_path, self.data_idx
        )

        max_retries = 3
        attempts = 0
        response = []

        while not response and attempts < max_retries:
            try:
                response = get_response(
                    final_answer_prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    seed=self.seed,
                    max_length=self.max_length,
                    truncation=self.truncation,
                    do_sample=self.do_sample,
                    max_new_tokens=self.max_new_tokens,
                    num_return_sequences=1,
                )
            except Exception as e:
                logging.error(
                    f"Error getting model response (attempt {attempts + 1}): {str(e)}"
                )
            attempts += 1

        if not response:
            logging.error("Failed to get reflection after all retries")
            return

            # Process response
        try:
            if (
                isinstance(response, list)
                and response
                and isinstance(response[0], dict)
            ):
                proposal = response[0].get("content", "")

            elif (
                isinstance(response, list) and response and isinstance(response[0], str)
            ):
                proposal = response[0]
            else:
                logging.error("Invalid output format of reflection, not a list")
                return

            final_answer = proposal

        except Exception as e:
            logging.error(f"Error processing model response: {str(e)}")
            return

        result = {
            "original query": self.data["question"],
            "last_query": current_node.state,
            "answer": final_answer,
            "path": path_q,
            "facts": knowledge,
        }
        return result

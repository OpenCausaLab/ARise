import json
import logging
import os
from datetime import datetime

import numpy as np
from tqdm import tqdm

from src.base import ToT_task, baseline_task
from src.task import MCTSTask
from utils.verify import F1_score_compute, exact_match

DATA_PATH = "./data/Wiki/data_s1.json"


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("debug.log_gpu0"),
            logging.StreamHandler(),
        ],
    )


def load_dataset(path):
    try:
        with open(path, encoding="utf-8") as f:
            dataset = json.load(f)
        return dataset
    except FileNotFoundError:
        logging.error(f"Dataset file not found at {path}")
        return []
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON format in dataset file at {path}")
        return []
    except Exception as e:
        logging.error(f"An error occurred while loading the dataset: {e}")
        return []


def main():
    # Define these at the start of main
    cleanup_model = None
    try:
        logging.info("Starting MCTS reasoning")

        # Initialize model
        from utils.inference_model import cleanup_model, initialize_model

        success = initialize_model()
        if not success:
            raise RuntimeError("Failed to initialize model")

        logging.info(f"Loading dataset from {DATA_PATH}")

        dataset = load_dataset(DATA_PATH)

        logging.info(f"Successfully loaded dataset with {len(dataset)} examples")

        total_count = 0
        correct_count = 0
        pass_N_count = 0
        F1_scores = []
        pass_N_F1_scores = []
        all_values = []

        # Process examples
        for idx in tqdm(range(len(dataset))):
            data = dataset[idx]

            logging.info(f"Processing example {idx + 1}/{len(dataset)}")

            output_dir = os.path.join("outputs", f"example_{idx + 1}")
            os.makedirs(output_dir, exist_ok=True)

            # Create MCTS task on current GPU
            task = MCTSTask(
                time_limit=None,  # time limit
                iteration_limit=30,  # iteration limit
                exploration_constant=1.0,  # Standard UCT constant
                multihops=5,  # sub query
                total_depth=4,
                data=data,  # Input question
                data_path=DATA_PATH,
                data_idx=idx,
                alpha=0.1,  # Value update rate
                temperature=0.7,  # Generation temperature
                ans_weight=1,  # ans weight
                max_tokens=2048,  # Max context tokens
                seed=170,  # Random seed
                do_sample=True,  # Enable sampling
                max_new_tokens=1024,  # Max new tokens
                low=0,  # Min value
                high=1,  # Max value
                run_mode="MCTS",
                value_mode="risk",
                value_model="Qwen7",
            )

            gt_facts = []
            for ele in data["supporting_facts"]:
                keyword = ele[0]
                fact_index = ele[1]
                for context in data["context"]:
                    if context[0] == keyword:
                        cont = context[1][fact_index]
                        gt_facts.append(keyword.strip() + ": " + cont.strip())
                        break
            gt_facts = "\n".join(gt_facts)

            # gt_facts = []
            # for ele in data["supporting_facts"]:
            #     keyword = list(ele.keys())[0]
            #     fact = ele[keyword]
            #     gt_facts.append(keyword.strip() + ": " + fact.strip())
            # gt_facts = "\n".join(gt_facts)

            try:
                if task.run_mode == "MCTS+RAG" or task.run_mode == "MCTS":
                    root_node = task.run()

                    # Save the search tree
                    tree_file = os.path.join(
                        output_dir,
                        "all_leaf_nodes_" + task.run_mode + ".json",
                    )
                    if os.path.exists(tree_file):
                        os.remove(tree_file)
                    leaf_nodes = task.traverse_tree(root_node)

                    best_leaf_file = os.path.join(
                        output_dir,
                        "best_leaf_" + task.run_mode + ".json",
                    )
                    if os.path.exists(best_leaf_file):
                        os.remove(best_leaf_file)
                    best_leaf = task.get_best_path()

                    for node in leaf_nodes:
                        all_values.append(node["value"])
                        if exact_match(node["answer"], data["answer"]):
                            node["correct"] = True
                        else:
                            node["correct"] = False
                        pass_N_F1_scores.append(
                            F1_score_compute(gt_facts, node["facts"])
                        )
                    for node in leaf_nodes:
                        if node["correct"]:
                            pass_N_count += 1
                            break

                    if exact_match(best_leaf["answer"], data["answer"]):
                        correct_count += 1
                        best_leaf["correct"] = True
                    else:
                        best_leaf["correct"] = False
                    total_count += 1

                    F1_scores.append(F1_score_compute(gt_facts, best_leaf["facts"]))

                    with open(best_leaf_file, "w") as f:
                        json.dump(best_leaf, f, indent=4)

                    with open(tree_file, "w") as f:
                        json.dump(leaf_nodes, f, indent=4)

                elif task.run_mode == "zero-shot":
                    solution = baseline_task(task, task.run_mode)
                    if exact_match(solution["answer"], data["answer"]):
                        correct_count += 1
                        solution["correct"] = True
                    else:
                        solution["correct"] = False
                    total_count += 1

                    output_file = os.path.join(output_dir, task.run_mode + ".json")
                    if os.path.exists(output_file):
                        os.remove(output_file)

                    F1_scores.append(F1_score_compute(gt_facts, solution["facts"]))

                    with open(output_file, "w") as f:
                        json.dump(solution, f, indent=4)

                elif task.run_mode == "tot":
                    solution = ToT_task(task)

                    if exact_match(solution["answer"], data["answer"]):
                        correct_count += 1
                        solution["correct"] = True
                    else:
                        solution["correct"] = False
                    total_count += 1

                    output_file = os.path.join(output_dir, task.run_mode + ".json")
                    if os.path.exists(output_file):
                        os.remove(output_file)

                    F1_scores.append(F1_score_compute(gt_facts, solution["facts"]))

                    with open(output_file, "w") as f:
                        json.dump(solution, f, indent=4)

                else:
                    raise ValueError(f"Invalid run mode: {task.run_mode}")

            except Exception as e:
                logging.error(f"Error processing example {idx + 1}: {e}")
                continue

        em_accuracy = correct_count / total_count if total_count > 0 else 0
        pass_N_em_accuracy = pass_N_count / total_count if total_count > 0 else 0

        F1_scores = np.array(F1_scores)
        F1_mean = np.mean(F1_scores)

        if pass_N_F1_scores is not None:
            pass_N_F1_scores = np.array(pass_N_F1_scores)
            pass_N_F1_mean = np.mean(pass_N_F1_scores)
        else:
            pass_N_F1_mean = 0

        all_values = np.array(all_values)
        all_values_mean = np.mean(all_values)

        logging.info(f"Total examples processed: {total_count}\n")
        logging.info(f"Correct predictions: {correct_count}\n")
        logging.info(f"Final Accuracy: {em_accuracy:.2%}\n")
        logging.info(f"Final F1 Score: {F1_mean:.2%}\n")
        logging.info(f"Pass N Accuracy: {pass_N_em_accuracy:.2%}\n")
        logging.info(f"Pass N F1 Score: {pass_N_F1_mean:.2%}\n")
        logging.info(f"avg_value: {all_values_mean:.2%}\n")

        time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        with open(f"results/{task.run_mode}_{time}.json", "w") as f:
            f.write(f"em_accuracy: {em_accuracy:.2%}\n")
            f.write(f"F1_score: {F1_mean:.2%}\n")
            f.write(f"pass_N_em_accuracy: {pass_N_em_accuracy:.2%}\n")
            f.write(f"pass_N_F1_score: {pass_N_F1_mean:.2%}\n")
            f.write(f"value_function: {task.value_mode}\n")
            f.write(f"value_model: {task.value_model}\n")
            f.write(f"total_examples: {total_count}\n")
            f.write(f"correct_predictions: {correct_count}\n")
            f.write(f"avg_value: {all_values_mean:.2%}\n\n")
            # record the experiment settings
            f.write(f"run_mode: {task.run_mode}\n")
            f.write(f"time_limit: {task.time_limit}\n")
            f.write(f"iteration_limit: {task.iteration_limit}\n")
            f.write(f"limit_type: {task.limit_type}\n")
            f.write(f"dataset: {DATA_PATH}\n")
            f.write(f"multihops: {task.multihops}\n")
            f.write(f"total_depth: {task.total_depth}\n")
            f.write(f"ans_weight: {task.ans_weight}\n")

    except Exception as e:
        logging.error(f"An error occurred in main: {e}")
    finally:
        if cleanup_model:  # Only call if it was successfully imported
            cleanup_model()


if __name__ == "__main__":
    setup_logging()
    main()

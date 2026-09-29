import copy
import json
import math
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.domain_data_manager import DomainDataManager


CONFIG_PATH = os.path.join(
    PROJECT_ROOT, "configs", "DGIL", "core50_20pct", "slca_dgil_20pct.json"
)
RUN_SEEDS = (1994, 1995, 1996)


def load_config():
    with open(CONFIG_PATH, encoding="utf-8") as config_file:
        return json.load(config_file)


def build_manager(base_args, run_seed, keep_ratio=True):
    args = copy.deepcopy(base_args)
    args["seed"] = run_seed
    if not keep_ratio:
        args.pop("train_domain_class_keep_ratio", None)
        args.pop("train_subset_seed", None)
    return DomainDataManager(
        args["dataset"],
        args["shuffle"],
        run_seed,
        args["init_cls"],
        args["increment"],
        args,
    )


def per_domain_paths(manager):
    result = {}
    for domain_name, stats in manager.train_domain_class_subset_stats.items():
        result[domain_name] = {
            int(class_id): set()
            for class_id in stats["before"]
        }

    for path, class_id, domain_id in zip(
        manager._original_train_data,
        manager._original_train_targets,
        manager._original_train_domain_idx,
    ):
        original_class_id = int(manager._class_order[int(class_id)])
        domain_name = manager.domain_names[int(domain_id)]
        result[domain_name][original_class_id].add(str(path))
    return result


def assert_ratio(manager):
    for domain_name, stats in manager.train_domain_class_subset_stats.items():
        for class_id, before in stats["before"].items():
            expected = math.floor(before * 0.2)
            actual = stats["after"][class_id]
            if actual != expected:
                raise AssertionError(
                    f"{domain_name}/class{class_id}: expected {expected}, got {actual}"
                )


def assert_nonempty_tasks(manager):
    for task_id in range(manager.nb_tasks):
        low = sum(manager._increments[:task_id])
        high = sum(manager._increments[: task_id + 1])
        count = int(np.count_nonzero((manager._train_targets >= low) & (manager._train_targets < high)))
        if count == 0:
            raise AssertionError(f"Task {task_id} has no training samples after strict floor sampling.")
        if count < 5:
            raise AssertionError(
                f"Task {task_id} has only {count} samples; S-Prompt KMeans requires at least 5."
            )


def main():
    args = load_config()
    baseline = build_manager(args, RUN_SEEDS[0], keep_ratio=False)
    baseline_test_paths = set(map(str, baseline._test_data.tolist()))

    managers = [build_manager(args, seed, keep_ratio=True) for seed in RUN_SEEDS]
    sampled_paths = []
    class_orders = []
    reference_domains = []

    for manager in managers:
        assert_ratio(manager)
        assert_nonempty_tasks(manager)
        if set(map(str, manager._test_data.tolist())) != baseline_test_paths:
            raise AssertionError("The 20% protocol changed the Core50 test image set.")
        sampled_paths.append(per_domain_paths(manager))
        class_orders.append(tuple(manager._class_order))
        reference_domains.append(tuple(int(value) for value in manager.ref_domain_ids))

    for current in sampled_paths[1:]:
        if current != sampled_paths[0]:
            raise AssertionError("The selected training image set changed across run seeds.")

    if len(set(class_orders)) != len(RUN_SEEDS):
        raise AssertionError("Class order did not vary across all configured run seeds.")
    if len(set(reference_domains)) != len(RUN_SEEDS):
        raise AssertionError("Reference-domain assignment did not vary across all configured run seeds.")

    total_before = sum(
        sum(stats["before"].values())
        for stats in managers[0].train_domain_class_subset_stats.values()
    )
    total_after = sum(
        sum(stats["after"].values())
        for stats in managers[0].train_domain_class_subset_stats.values()
    )
    print("Core50 20% DGIL data audit passed.")
    print(f"Training pool: {total_before} -> {total_after}")
    print(f"Test images unchanged: {len(baseline_test_paths)}")
    for seed, manager in zip(RUN_SEEDS, managers):
        print(
            f"seed={seed} class_order_head={manager._class_order[:10]} "
            f"reference_domains={list(map(int, manager.ref_domain_ids))}"
        )


if __name__ == "__main__":
    main()

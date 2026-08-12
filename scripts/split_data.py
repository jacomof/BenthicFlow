"""Split the normalized data into train/val/test CSVs."""

import csv
import random
from pathlib import Path

seed = 42


def iter_files(deployment_path: Path):
    for path in deployment_path.rglob("*"):
        if path.is_file():
            yield path


def write_split_csv(rows, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["key", "campaign", "deployment", "path"])
        writer.writerows(rows)


def build_rows(deployment_path: Path, campaign_name: str):
    rows = []
    for file_path in iter_files(deployment_path):
        rows.append(
            [
                file_path.stem,
                campaign_name,
                deployment_path.name,
                str(file_path),
            ]
        )
    return rows


if __name__ == "__main__":
    random.seed(seed)
    data_path = Path("./data_normalized")

    train_rows = []
    val_rows = []
    test_rows = []

    # --- val ---
    print("Selecting val deployments...")
    selected_deployments_val = []
    for dir in data_path.iterdir():
        deployments = list(dir.iterdir())
        if not deployments:
            raise ValueError(f"No deployments available for val selection in {dir}")
        selected_deployment = deployments[random.randint(0, len(deployments) - 1)]
        print(f"Selected val deployment from {dir.name}: {selected_deployment}")
        selected_deployments_val.append(selected_deployment)
        val_rows.extend(build_rows(selected_deployment, dir.name))
        print("Finished collecting val rows.")

    # --- test ----
    print("Selecting test deployments...")
    selected_deployments_test = []
    for dir in data_path.iterdir():
        deployments = list(dir.iterdir())
        deployments = [d for d in deployments if d not in selected_deployments_val]
        if not deployments:
            raise ValueError(f"No deployments available for test selection in {dir}")
        selected_deployment = deployments[random.randint(0, len(deployments) - 1)]
        print(f"Selected test deployment from {dir.name}: {selected_deployment}")
        selected_deployments_test.append(selected_deployment)
        test_rows.extend(build_rows(selected_deployment, dir.name))
        print("Finished collecting test rows.")

    # --- train ---
    print("Selecting training deployments...")
    for dir in data_path.iterdir():
        deployments = list(dir.iterdir())
        deployments = [
            d
            for d in deployments
            if d not in selected_deployments_val and d not in selected_deployments_test
        ]
        print(
            f"Selected training deployments from {dir.name}: {[d.name for d in deployments]}"
        )
        for selected_deployment_train in deployments:
            train_rows.extend(build_rows(selected_deployment_train, dir.name))
    print("Finished collecting training rows.")

    output_dir = Path("./data_split")
    write_split_csv(train_rows, output_dir / "train.csv")
    write_split_csv(val_rows, output_dir / "val.csv")
    write_split_csv(test_rows, output_dir / "test.csv")
    print(f"Wrote CSVs to: {output_dir}")

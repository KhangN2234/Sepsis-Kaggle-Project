"""
Render XGBoost tree structure from the tuned model.

Usage:
    .\\.venv\\Scripts\\python.exe graph.py

Outputs:
    tuning_results\\trees\\tree_0.png
    tuning_results\\trees\\tree_1.png
    ...
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from joblib import load as joblib_load


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _build_tree_index(tree_df):
    """Build parent/child relationships for a single tree dataframe."""
    nodes = {}
    for _, row in tree_df.iterrows():
        node_id = str(row["ID"])
        nodes[node_id] = row.to_dict()
        nodes[node_id]["children"] = []

    for node_id, row in nodes.items():
        if row.get("Feature") == "Leaf":
            continue
        for child_key in ("Yes", "No"):
            child_id = row.get(child_key)
            if child_id is not None and str(child_id) in nodes:
                row["children"].append(str(child_id))
    return nodes


def _assign_positions(nodes, root_id):
    """Assign x/y positions to nodes using a simple recursive layout."""
    positions = {}
    leaf_cursor = [0]

    def walk(node_id, depth):
        node = nodes[node_id]
        children = node["children"]
        if not children:
            x = leaf_cursor[0]
            leaf_cursor[0] += 1
            positions[node_id] = (x, -depth)
            return x

        child_xs = [walk(child_id, depth + 1) for child_id in children]
        x = sum(child_xs) / len(child_xs)
        positions[node_id] = (x, -depth)
        return x

    walk(root_id, 0)
    return positions


def _format_node_label(node_row):
    feature = node_row.get("Feature", "")
    if feature == "Leaf":
        return f"Leaf\nvalue={node_row.get('Gain', 0):.4f}\ncover={node_row.get('Cover', 0):.1f}"

    split = node_row.get("Split", "")
    gain = node_row.get("Gain", 0)
    cover = node_row.get("Cover", 0)
    return f"{feature} < {split}\ngain={gain:.3f}\ncover={cover:.1f}"


def _render_single_tree(tree_df, out_path: Path, title: str) -> None:
    nodes = _build_tree_index(tree_df)
    root_candidates = [node_id for node_id, row in nodes.items() if str(row.get("ID")) == node_id]
    root_id = None
    for candidate in root_candidates:
        if nodes[candidate].get("Node", 0) == 0:
            root_id = candidate
            break
    if root_id is None:
        root_id = min(nodes.keys(), key=lambda value: int(str(value).split("-")[-1]) if str(value).split("-")[-1].isdigit() else 0)

    positions = _assign_positions(nodes, root_id)

    fig, ax = plt.subplots(figsize=(24, 12))
    ax.set_title(title)
    ax.axis("off")

    # Draw edges first.
    for node_id, row in nodes.items():
        x0, y0 = positions[node_id]
        for child_id in row["children"]:
            x1, y1 = positions[child_id]
            ax.plot([x0, x1], [y0, y1], color="gray", linewidth=1.2)
            mid_x = (x0 + x1) / 2
            mid_y = (y0 + y1) / 2
            ax.text(mid_x, mid_y, "yes" if child_id == row.get("Yes") else "no", fontsize=8, color="darkred")

    # Draw nodes.
    for node_id, row in nodes.items():
        x, y = positions[node_id]
        label = _format_node_label(row)
        width = 0.95
        height = 0.6 if row.get("Feature") != "Leaf" else 0.5
        patch = FancyBboxPatch(
            (x - width / 2, y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.08",
            linewidth=1.2,
            edgecolor="navy" if row.get("Feature") != "Leaf" else "darkgreen",
            facecolor="white",
        )
        ax.add_patch(patch)
        ax.text(x, y, label, ha="center", va="center", fontsize=8)

    ax.relim()
    ax.autoscale_view()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_trees(model_path: Path, output_dir: Path, max_trees: int = 3) -> None:
    """Load a saved XGBoost model and render a few trees as PNG files."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    model = joblib_load(model_path)
    booster = model.get_booster() if hasattr(model, "get_booster") else model

    tree_count = booster.num_boosted_rounds() if hasattr(booster, "num_boosted_rounds") else max_trees
    trees_to_render = min(max_trees, tree_count)

    for tree_index in range(trees_to_render):
        out_path = output_dir / f"tree_{tree_index}.png"
        tree_df = booster.trees_to_dataframe()
        single_tree_df = tree_df[tree_df["Tree"] == tree_index].copy()
        _render_single_tree(single_tree_df, out_path, f"XGBoost Tree {tree_index}")
        print(f"Saved {out_path}")

    # Also save a text dump for easier inspection.
    dump_path = output_dir / "tree_dump.txt"
    with open(dump_path, "w", encoding="utf-8") as f:
        f.write(booster.get_dump(dump_format="text")[0])
    print(f"Saved {dump_path}")


def main() -> None:
    model_path = Path("tuning_results") / "final_model.pkl"
    output_dir = Path("tuning_results") / "trees"
    render_trees(model_path=model_path, output_dir=output_dir, max_trees=3)


if __name__ == "__main__":
    main()

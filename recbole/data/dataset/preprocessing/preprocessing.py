"""
preprocess.py

Flexible preprocessing for JSONL datasets.
Author: Piotr Stachowicz
"""

import json
from collections.abc import Callable
from typing import List, Set
import argparse


def create_id_mapping(ids: Set[str]) -> tuple[dict[str, int], dict[int, str]]:
    """Map string IDs to integer IDs."""
    return (
        {id: idx for idx, id in enumerate(ids)}, 
        {idx: id for idx, id in enumerate(ids)}
    )


def save_mapping(mapping: dict, path: str) -> None:
    """Save the mapping dictionary as a JSON file."""
    with open(path, 'w') as f:
        for key, value in mapping.items():
            json.dump({key: value}, f)
            f.write('\n')


def get_valid_ids(
    path: str,
    column: str,
    condition: Callable[[int], bool]
) -> Set[str]:
    """Return a set of IDs from a given column that satisfy the condition."""
    counts = {}

    with open(path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            key = entry[column]
            counts[key] = counts.get(key, 0) + 1

    return {k for k, v in counts.items() if condition(v)}


def save_inter(
    path: str,
    filters: List[Callable[[dict], bool]],
    user_mapping: dict,
    item_mapping: dict
) -> None:
    """Export filtered interactions from JSONL to TSV format for RecBole."""
    output_file = f"{path.rsplit('/', 1)[-1].split('.')[0]}.inter"

    with open(path, 'r') as f, open(output_file, 'w') as out:
        out.write('user_id:token\titem_id:token\trating:float\ttimestamp:float\n')

        for line in f:
            entry = json.loads(line)

            if not all(f(entry) for f in filters):
                continue

            user_id = user_mapping[entry['user_id']]
            product_id = item_mapping[entry['parent_asin']]
            rating = entry['rating']
            timestamp = entry['timestamp']

            out.write(
                f"{user_id}\t{product_id}\t{rating}\t{timestamp}\n"
            )


def save_item(
    path: str,
    filters: List[Callable[[dict], bool]],
    item_mapping: dict
) -> None:
    """Export filtered items from JSONL to TSV format for RecBole."""
    output_file = f"{path.rsplit('/', 1)[-1].split('.')[0]}.item"
    
    with open(path, 'r') as f, open(output_file, 'w') as out:
        out.write(
            "item_id:token\ttitle:token\tprice:float\tsales_type:token\tsales_rank:float\tbrand:token\tcategories:token_seq\n"
        )

        for line in f:
            entry = json.loads(line)

            if not all(f(entry) for f in filters):
                continue

            product_id = item_mapping[entry['parent_asin']]
            title = entry['title']
            price = entry['price']
            sales_type = entry['main_category']
            sales_rank = entry['details'].get('Best Sellers Rank', {}).get(sales_type, 'null')
            brand = entry['store']
            categories_table = entry['categories']

            categories = ""

            if len(categories) >= 2:
                for cat in categories_table[:len(categories_table) - 2]:
                    categories += f"\'{cat}\', "

            if categories_table:
                categories += f"\'{categories_table[-1]}\'"

                out.write(
                    f"{product_id}\t{title}\t{price}\t{sales_type}\t{sales_rank}\t\'{brand}\'\t{categories}\n"
                )
            else:
                out.write(
                    f"{product_id}\t{title}\t{price}\t{sales_type}\t{sales_rank}\t\'{brand}\'\n"
                )


def preprocess(path: str) -> None:
    """Main preprocessing pipeline."""
    # Config
    min_user_interactions = 5
    min_item_interactions = 0

    valid_users = get_valid_ids(path, 'user_id', lambda c: c >= min_user_interactions)

    valid_items = get_valid_ids(path, 'parent_asin', lambda c: c >= min_item_interactions)
    
    # Mappings
    user_mapping, reverse_user_mapping = create_id_mapping(valid_users)
    item_mapping, reverse_item_mapping = create_id_mapping(valid_items)

    file_name = path.rsplit('/', 1)[-1].split('.')[0]
    prefix_path = path.split(file_name)[0]
    
    save_mapping(user_mapping, f'{prefix_path}user_mapping_{file_name}.jsonl')
    save_mapping(item_mapping, f'{prefix_path}item_mapping_{file_name}.jsonl')
    save_mapping(reverse_user_mapping, f'{prefix_path}reverse_user_mapping_{file_name}.jsonl')
    save_mapping(reverse_item_mapping, f'{prefix_path}reverse_item_mapping_{file_name}.jsonl')

    # Predicates
    preds1 = [
        lambda entry: entry['user_id'] in valid_users,
        lambda entry: entry['parent_asin'] in valid_items
    ]

    preds2 = [
        lambda entry: entry['parent_asin'] in valid_items
    ]

    save_inter(path, preds1, user_mapping, item_mapping)
    save_item(f'{prefix_path}meta_{file_name}.jsonl', preds2, item_mapping)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', required=True)

    args = parser.parse_args()

    path = args.path

    preprocess(path)

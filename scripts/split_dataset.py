import json
import random
import argparse
import os

def split_jsonl(input_file, train_file, test_file, test_ratio=0.2, seed=42):
    """
    Reads a JSONL file, shuffles the data, and splits it into train and test sets.

    Each line is expected to be a valid JSON object with keys such as:
    - "query"
    - "context"
    - "response"
    - Optionally "node_ids"

    Args:
        input_file (str): Path to the input JSONL file.
        train_file (str): Path where the training JSONL file will be written.
        test_file (str): Path where the test JSONL file will be written.
        test_ratio (float): Fraction of data to use for the test split.
        seed (int): Random seed for reproducible shuffling.
    """
    # Read and parse the JSONL file
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    data = [json.loads(line) for line in lines]

    # Shuffle the data to randomize the order
    random.seed(seed)
    random.shuffle(data)

    # Compute the split index based on test_ratio
    n_total = len(data)
    n_test = int(n_total * test_ratio)
    test_data = data[:n_test]
    train_data = data[n_test:]

    def write_jsonl(data, output_file):
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

    write_jsonl(train_data, train_file)
    write_jsonl(test_data, test_file)

    print(f"Total examples: {n_total}")
    print(f"Train examples: {len(train_data)}")
    print(f"Test examples: {len(test_data)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Split a JSONL file into train and test splits."
    )
    parser.add_argument(
        "--input_file", type=str, default="data/retrieval_qa.jsonl",
        help="Path to the input JSONL file (default: data/retrieval_qa.jsonl)"
    )
    parser.add_argument(
        "--train_file", type=str, default="data/retrieval_qa_train.jsonl",
        help="Path to the output training JSONL file (default: data/retrieval_qa_train.jsonl)"
    )
    parser.add_argument(
        "--test_file", type=str, default="data/retrieval_qa_test.jsonl",
        help="Path to the output test JSONL file (default: data/retrieval_qa_test.jsonl)"
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.2,
        help="Fraction of data to use as test split (default: 0.2)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for shuffling (default: 42)"
    )
    args = parser.parse_args()

    # Create output directories if they don't exist
    for file_path in [args.train_file, args.test_file]:
        output_dir = os.path.dirname(file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

    split_jsonl(args.input_file, args.train_file, args.test_file, args.test_ratio, args.seed)
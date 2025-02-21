import json
import argparse
import os

def transform_example(example: dict) -> dict:
    """
    Transform a raw retrieval QA example into a conversational format with enhanced instructions.
    
    Input Example (from retrieval_qa.jsonl):
        {
            "query": "Where is the recliner sofa in the area?",
            "context": "Available Locations for Selection: ...",
            "response": "area_5_cluster_0"
        }
    
    Output Example:
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Query: Where is the recliner sofa in the area?\nContext: Available Locations for Selection: ...\nInstruction: From the provided context, please choose exactly one option and respond with the name exactly as it appears (one word)."
                },
                {
                    "role": "assistant",
                    "content": "Answer: area_5_cluster_0"
                }
            ]
        }
    """
    user_message = (
        f"Query: {example.get('query', '')}\n"
        f"Context: {example.get('context', '')}\n"
        "Instruction: From the provided context, please choose exactly one option and respond with the name exactly as it appears."
    )
    
    messages = [
        {
            "role": "user",
            "content": user_message
        },
        {
            "role": "assistant",
            "content": f"{example.get('response', '')}"
        }
    ]
    return {"messages": messages}

def transform_file(input_file: str, output_file: str):
    """
    Read the input JSONL file, transform each example into the conversational format, and write to the output JSONL file.
    """
    with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                example = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {line}")
                continue
            transformed = transform_example(example)
            fout.write(json.dumps(transformed) + "\n")
    print(f"Transformed data written to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transform retrieval QA JSONL file to a conversational format with enhanced instructions."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="data/retrieval_qa.jsonl",
        help="Path to the input retrieval QA JSONL file (default: data/retrieval_qa.jsonl)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/retrieval_qa_conversational.jsonl",
        help="Path to the output conversational JSONL file (default: data/retrieval_qa_conversational.jsonl)"
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    transform_file(args.input_file, args.output_file) 
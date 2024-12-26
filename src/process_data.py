from pathlib import Path

import typer
from bs4 import BeautifulSoup
from datasets import load_dataset

app = typer.Typer()
# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJ_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"


@app.command()
def preprocess_dataset(
        output_path: Path = PROCESSED_DATA_DIR,
        dataset_name: str = "ymoslem/Law-StackExchange",
):
    """Load, preprocess, and save the dataset."""
    dataset = load_dataset(dataset_name)['train']

    def clean_html(text):
        return BeautifulSoup(text, 'html.parser').get_text()

    def preprocess(example):
        example["question_body"] = f"{example['question_title']} \\n" \
                                   f"{clean_html(example['question_body'])}"

        if example['answers']:
            example['answers'] = [
                {
                    "answer_id": answer['answer_id'],
                    "body": f"{clean_html(answer['body'])}",
                    "score": answer['score']
                } for answer in example['answers']
            ]

        return example

    processed_data = dataset.map(preprocess)

    processed_data = processed_data.remove_columns(['question_title', 'link', 'license', 'tags'])
    processed_data = processed_data.rename_column("question_body",
                                                  "question")

    processed_data.save_to_disk(output_path)


if __name__ == "__main__":
    preprocess_dataset()

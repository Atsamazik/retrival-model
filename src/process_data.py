from pathlib import Path

import typer
from rich import print
from rich.console import Console
from rich.table import Table
from rich.pretty import Pretty
from bs4 import BeautifulSoup
from datasets import load_dataset, load_from_disk

app = typer.Typer(name="Data process CLI")

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


@app.command()
def show_saved_data(number: int = 2):
    """Show saved data in a readable format."""
    dataset = load_from_disk(PROCESSED_DATA_DIR)
    total_rows = len(dataset)
    console = Console()

    console.rule("[bold green]Dataset Overview[/bold green]")
    console.print(
        Pretty(
            {
                "features": dataset.features.keys(),
                "num_rows": total_rows,
            },
            expand_all=True,
        )
    )
    console.print("\n")

    print(f"[bold green]Dataset contains {total_rows} rows.[/bold green]\n")

    for idx in range(min(number, total_rows)):
        entry = dataset[idx]
        console.rule(f"[bold blue]Entry {idx + 1}[/bold blue]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Key", style="dim")
        table.add_column("Value", overflow="fold")

        for key, value in entry.items():
            table.add_row(str(key), str(value))

        console.print(table)
        console.print("\n")


if __name__ == "__main__":
    app()

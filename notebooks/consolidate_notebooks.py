import json
import os

def strip_outputs(notebook_path):
    """Load a notebook and strip outputs from code cells."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb.get('cells', []):
        if cell['cell_type'] == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None

    return nb

def main():
    # List of notebooks in order
    notebook_files = [
        '01_preprocessing.ipynb',
        '02_segmentation.ipynb',
        '03_cdr_calculation.ipynb',
        '04_training.ipynb',
        '05_evaluation.ipynb',
        '06_final_output.ipynb'
    ]

    # Base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Read the first notebook to get metadata and nbformat
    first_nb_path = os.path.join(base_dir, notebook_files[0])
    with open(first_nb_path, 'r', encoding='utf-8') as f:
        first_nb = json.load(f)

    # Create combined notebook structure
    combined_nb = {
        "cells": [],
        "metadata": first_nb.get('metadata', {}),
        "nbformat": first_nb.get('nbformat', 4),
        "nbformat_minor": first_nb.get('nbformat_minor', 4)
    }

    # Add an introductory markdown cell
    intro_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Glaucoma Detection Project - Consolidated Notebook\n",
            "\n",
            "This notebook is automatically generated from the sub-notebooks:\n",
            "- 01_preprocessing.ipynb\n",
            "- 02_segmentation.ipynb\n",
            "- 03_cdr_calculation.ipynb\n",
            "- 04_training.ipynb\n",
            "- 05_evaluation.ipynb\n",
            "- 06_final_output.ipynb\n",
            "\n",
            "**Note:** Outputs have been stripped from code cells to reduce file size and avoid redundancy.\n",
            "To generate outputs, run the notebook cells.\n",
            "\n",
            "---\n"
        ]
    }
    combined_nb["cells"].append(intro_cell)

    # Process each notebook
    for nb_file in notebook_files:
        nb_path = os.path.join(base_dir, nb_file)
        if not os.path.exists(nb_path):
            print(f"Warning: Notebook {nb_file} not found. Skipping.")
            continue

        print(f"Processing {nb_file}...")
        nb = strip_outputs(nb_path)
        combined_nb["cells"].extend(nb.get('cells', []))

    # Write the combined notebook
    output_path = os.path.join(base_dir, 'glaucoma_project_consolidated.ipynb')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(combined_nb, f, indent=1)

    print(f"\nConsolidated notebook created: {output_path}")
    print("To update the consolidated notebook after changes to sub-notebooks, run this script again.")

if __name__ == "__main__":
    main()
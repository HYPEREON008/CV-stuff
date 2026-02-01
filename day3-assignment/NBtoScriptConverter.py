import json

notebook_file = "dogs_cats_classification.ipynb"

train_file = "train.py"
eval_file = "evaluate.py"

# Marker heading
split_heading = "## Evaluation on Test Set"

with open(notebook_file, "r", encoding="utf-8") as f:
    nb = json.load(f)

before_split = True

with open(train_file, "w", encoding="utf-8") as train_out, \
     open(eval_file, "w", encoding="utf-8") as eval_out:

    for cell in nb["cells"]:

        # Check markdown split point
        if cell["cell_type"] == "markdown":
            text = "".join(cell["source"])

            if split_heading in text:
                before_split = False
                continue  # don't write heading itself anywhere

        # Only code cells get exported
        if cell["cell_type"] == "code":
            code = "".join(cell["source"]).strip()

            if code:  # skip empty code cells
                if before_split:
                    train_out.write(code + "\n\n")
                else:
                    eval_out.write(code + "\n\n")

print("Training code saved to:", train_file)
print("Evaluation code saved to:", eval_file)

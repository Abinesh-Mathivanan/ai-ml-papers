import os

def load_text_data(data_dir="data"):
    texts = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as file:
                texts.append(file.read())
    return texts
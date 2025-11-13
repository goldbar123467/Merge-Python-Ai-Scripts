# merge_curiosity_cnn.py
from datasets import load_from_disk

cnn = load_from_disk("./cnn_classification")
curiosity = load_from_disk("./curiosity_dataset")

def add_cnn_insight(example):
    text = example.get("text", "").lower()
    if any(word in text for word in ["election", "trump", "biden", "vote"]):
        politics = cnn.filter(lambda x: x["label"] == 4)  # 4 = politics
        if len(politics) > 0:
            example["cnn_insight"] = politics[0]["text"][:200] + "..."
    return example

merged = curiosity.map(add_cnn_insight)
merged.save_to_disk("./curiosity_with_cnn")
print("MERGED: Curiosity + CNN Politics Insight")
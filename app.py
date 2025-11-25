import json
from sentence_transformers import SentenceTransformer
import os

def main():
    # Load MedEmbed model
    model = SentenceTransformer("abhinand/MedEmbed-base-v0.1")

    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)

    while True:
        text = input("Enter text to embed: ").strip()
        if not text:
            continue

        vector = model.encode(text).tolist()

        data = {
            "input": text,
            "vector": vector
        }

        # Append as one-line JSON object
        with open("output/embedding_output.jsonl", "a") as f:
            f.write(json.dumps(data) + "\n")

        print("Saved one line to embedding_output.jsonl")

if __name__ == "__main__":
    main()

import json
import chromadb
from sentence_transformers import SentenceTransformer

file_path = 'data.json'
with open(file_path, 'r') as file:
    data = json.load(file)

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

def get_embedding(text):
    return model.encode(text).tolist()

client = chromadb.Client() 

collection = client.create_collection(name="rag")

idx = 0
for entry in data:
    for mail in entry["mails"]:
        text = f"{mail['caller']}: {mail['description']}"
        embedding = get_embedding(text)
        metadata = {
            "caller": mail["caller"],
            "description": mail["description"]
        }
        collection.add(
            ids=[str(idx)],
            embeddings=[embedding],
            metadatas=[metadata]
        )
        idx += 1

print("Data inserted into ChromaDB successfully.")

query = "How did Company A respond about CXMAHO?"
query_embedding = get_embedding(query)

results = collection.query(query_embeddings=[query_embedding], n_results=3)


print("----------------------------------")
print("")
print("")
print(results)
print("")
print("")
print("----------------------------------")

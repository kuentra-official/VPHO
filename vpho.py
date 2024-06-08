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

collection = client.create_collection(name="vpho_rag")

idx = 0
for entry in data:
    metadata = {
        "logs": json.dumps(entry["mails"]),  
    }
    section_text = entry["section"]
    embedding = get_embedding(section_text)
    print(entry)
    collection.add(
        ids=[str(idx)],
        embeddings=[embedding],
        metadatas=[metadata]
    )
    idx += 1
    
print("Data inserted into ChromaDB successfully.")


query = "How is the discussion about CXMAHO with Company A going?"
query_embedding = get_embedding(query)

results = collection.query(query_embeddings=[query_embedding], n_results=1)

print("----------------------------------")
print("")
print("")
print(results)
print("")
print("")
print("----------------------------------")
import os
from openai import AzureOpenAI
import dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables from .env file
dotenv.load_dotenv()

client = AzureOpenAI(
  api_key = os.getenv("AZURE_OPENAI_API_KEY"),  
  api_version = "2024-06-01",
  azure_endpoint =os.getenv("AZURE_OPENAI_ENDPOINT") 
)

resp = client.embeddings.create(
    input = ["my cat is cute", "my kitty is adorable"],
    model= "text-embedding-3-large"
)

#print(resp.model_dump_json(indent=2))

embedding_a = resp.data[0].embedding
embedding_b = resp.data[1].embedding

print(f"embedding_a: {embedding_a[:15]}")
print(f"embedding_b: {embedding_b[:15]}")

similarity_score = np.dot(embedding_a, embedding_b)
print(f"dot product: {similarity_score}")

# Reshape embeddings to 2D arrays
embedding_a = np.array(embedding_a).reshape(1, -1)
embedding_b = np.array(embedding_b).reshape(1, -1)




# Calculate cosine similarity
cosine_similarity_score = cosine_similarity(embedding_a, embedding_b)

print(f"cosine similarity: {cosine_similarity_score[0][0]}")

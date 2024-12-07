from typing import Any, List 
from llama_index.core.bridge.pydantic import PrivateAttr 
from llama_index.core.embeddings import BaseEmbedding 
import requests 
import json 


# Custom function to fetch embeddings from an external API
def fetch_embeddings_from_api(text: str, model="nomic-embed-text"):
    url = "http://localhost:11434/api/embeddings"
    headers = {"Content-Type": "application/json"}
    payload = {"model": model, "prompt": text}
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            data = response.json()
            return data.get("embedding", [])
        else:
            print(f"API Error: {response.status_code}, {response.text}")
            return [0.0] * 384  # Return a dummy embedding if API fails
    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        return [0.0] * 384  # Return a dummy embedding if exception occurs

class InstructorEmbeddings(BaseEmbedding):
    _instruction: str = PrivateAttr()

    def __init__(
        self,
        instruction: str = "Represent a document for semantic search:",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._instruction = instruction

    @classmethod
    def class_name(cls) -> str:
        return "instructor_with_api"

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    def _get_query_embedding(self, query: str) -> List[float]:
        print(f"Getting query embedding for: {query}")
        embedding = fetch_embeddings_from_api(f"{self._instruction} {query}")
        return embedding

    def _get_text_embedding(self, text: str) -> List[float]:
        print(f"Getting text embedding for: {text}")
        embedding = fetch_embeddings_from_api(f"{self._instruction} {text}")
        return embedding

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        print(f"Getting embeddings for {len(texts)} texts")
        embeddings = [fetch_embeddings_from_api(f"{self._instruction} {text}") for text in texts]
        return embeddings

import os
import uuid
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
import numpy as np

class RagImageConverter:
    def __init__(self, db_dir="~/.ai_oracle/chroma_data", collection_name="image_snapshots"):
        self.db_dir = os.path.expanduser(db_dir)
        os.makedirs(self.db_dir, exist_ok=True)
        
        self.chroma_client = chromadb.PersistentClient(path=self.db_dir)
        self.embedding_function = OpenCLIPEmbeddingFunction()
        
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )

    def vectorize_and_save(self, image_rgb_np, description, time_str):
        image_id = str(uuid.uuid4())
        
        self.collection.add(
            ids=[image_id],
            images=[image_rgb_np],
            metadatas=[{"time_str": time_str, "description": description}],
        )

    def search_similar(self, image_rgb_np, top_k=2):
        if self.collection.count() == 0:
            return []
            
        results = self.collection.query(
            query_images=[image_rgb_np],
            n_results=min(top_k, self.collection.count())
        )
        
        similar_descriptions = []
        if results and 'metadatas' in results and results['metadatas']:
            for meta_list in results['metadatas']:
                for m in meta_list:
                    if m and "description" in m:
                        similar_descriptions.append(f"[{m.get('time_str', 'unknown')}] {m['description']}")
                        
        return similar_descriptions

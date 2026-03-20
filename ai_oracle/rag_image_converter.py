import os
import uuid
import chromadb
from datetime import datetime
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction

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
        
        # Add full ISO date for better chronological filtering in RAG
        now = datetime.now()
        iso_timestamp = now.isoformat()
        date_str = now.strftime("%Y-%m-%d")
        
        self.collection.add(
            ids=[image_id],
            images=[image_rgb_np],
            metadatas=[{
                "time_str": time_str, 
                "description": description,
                "timestamp": iso_timestamp,
                "date": date_str
            }],
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
                        # Return full date + time for better context
                        date_tag = m.get('date', '')
                        comp_ts = f"{date_tag} {m.get('time_str', 'unknown')}".strip()
                        similar_descriptions.append(f"[{comp_ts}] {m['description']}")
                        
        return similar_descriptions

    def search_by_text(self, query_text, top_k=8, date_filter=None):
        """Perform semantic vector search using the user's text prompt, optionally filtered by date."""
        if self.collection.count() == 0:
            return []
            
        where_params = None
        if date_filter:
            where_params = {"date": date_filter}
            
        results = self.collection.query(
            query_texts=[query_text],
            n_results=min(top_k, self.collection.count()),
            where=where_params
        )
        
        context_lines = []
        if results and 'metadatas' in results and results['metadatas']:
            for meta_list in results['metadatas']:
                for m in meta_list:
                    if m and "description" in m:
                        date_tag = m.get('date', '')
                        comp_ts = f"{date_tag} {m.get('time_str', '?')}".strip()
                        context_lines.append(f"[{comp_ts}] {m['description']}")
                        
        return context_lines

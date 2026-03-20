import os
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction

def clear_rag():
    db_dir = os.path.expanduser("~/.ai_oracle/chroma_data")
    collection_name = "image_snapshots"
    
    print(f"Connecting to ChromaDB at {db_dir}...")
    client = chromadb.PersistentClient(path=db_dir)
    
    try:
        print(f"Deleting collection '{collection_name}'...")
        client.delete_collection(name=collection_name)
        print("Collection deleted successfully!")
    except Exception as e:
        print(f"Collection '{collection_name}' might not exist or failed to delete: {e}")
    
    # Re-create it fresh if needed (optional since the app will do it, but good to confirm)
    try:
        print(f"Re-creating empty collection '{collection_name}'...")
        client.get_or_create_collection(
            name=collection_name,
            embedding_function=OpenCLIPEmbeddingFunction()
        )
        print("Empty collection ready!")
    except Exception as e:
        print(f"Error recreating collection: {e}")

if __name__ == "__main__":
    clear_rag()

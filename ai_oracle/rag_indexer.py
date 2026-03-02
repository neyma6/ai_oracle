import os
import argparse
from pathlib import Path
import chromadb
import ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

def get_text_files(root_dir):
    """
    Recursively walk through the directory tree from `root_dir` to find all relevant source code files.
    This filters out binary files, unneeded metadata, and build directories to keep the vector database clean.
    It currently defaults specifically to Java.
    """
    text_extensions = {
        ".java"
        # ".txt", ".md", ".py", ".json", ".csv", ".html",
        # ".js", ".ts", ".java", ".cpp", ".c", ".h", ".go", ".rs"
    }
    ignore_dirs = {".git", "venv", "__pycache__", ".idea", "node_modules", "ai_oracle.egg-info", "chroma_data", ".venv", "build", "gradle", "target"}
    
    files = []
    root_path = Path(root_dir).resolve()
    
    for path in root_path.rglob("*"):
        # Ignore irrelevant directories
        if any(ignored in path.parts for ignored in ignore_dirs):
            continue
            
        if path.is_file() and not path.name.startswith("."):
            if path.suffix.lower() in text_extensions or path.suffix == "":
                files.append(path)
                
    return files

def chunk_text(text, extension, chunk_size=1000, overlap=200):
    """
    Breaks large documents down into smaller pieces so they can fit inside context windows 
    and produce more focused embeddings.
    
    Instead of slicing code at random character counts (which might cut a class definition in half), 
    this uses LangChain's RecursiveCharacterTextSplitter. It tries to split intelligently 
    along natural boundaries like functions, classes, and newlines based on the language's AST.
    """
    lang_map = {
        ".java": Language.JAVA,
        ".py": Language.PYTHON,
        ".js": Language.JS,
        ".ts": Language.TS,
        ".go": Language.GO,
        ".cpp": Language.CPP,
        ".rs": Language.RUST,
        ".html": Language.HTML,
        ".md": Language.MARKDOWN,
    }
    
    lang = lang_map.get(extension.lower())
    
    if lang:
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=lang, chunk_size=chunk_size, chunk_overlap=overlap
        )
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=overlap
        )
        
    return splitter.split_text(text)

def main():
    """
    Main entry point for the indexing script.
    - Parses CLI paths (source code directory and database destination).
    - Checks that Ollama is currently running the correct embedding model locally.
    - Connects to (or creates) the SQLite-backed Chroma database.
    - Loops over every file, chunks the text, creates vectors, and saves it.
    """
    parser = argparse.ArgumentParser(description="Index files into a persistent vector database using Ollama + ChromaDB.")
    parser.add_argument("--root-dir", type=str, required=True, help="Root directory to parse")
    parser.add_argument("--db-dir", type=str, default=os.path.expanduser("~/.ai_oracle/chroma_data"), help="Directory to store the persistent Chroma vector DB")
    parser.add_argument("--model", type=str, default="nomic-embed-text", help="Ollama embedding model to use")
    parser.add_argument("--collection", type=str, default="documents", help="Chroma collection name")
    
    args = parser.parse_args()
    
    # Pre-flight check: ensure the user actually has the embedding model downloaded
    # and Ollama's daemon is running.
    print(f"Checking if Ollama model '{args.model}' is available locally...")
    try:
        ollama.show(args.model)
    except Exception:
        print(f"Warning: Model '{args.model}' not found in Ollama.")
        print(f"Consider running: ollama pull {args.model}")
        return

    from chromadb.utils import embedding_functions
    
    # We bind the vector embedding logic natively into ChromaDB.
    # This means whenever we insert text (or later query text), Chroma automatically calls 
    # out to the local Ollama instance on port 11434 to convert the string to a mathematical array.
    ollama_ef = embedding_functions.OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name=args.model,
    )

    print(f"Initializing persistent ChromaDB client at {args.db_dir}...")
    chroma_client = chromadb.PersistentClient(path=args.db_dir)
    collection = chroma_client.get_or_create_collection(name=args.collection, embedding_function=ollama_ef)
    
    files = get_text_files(args.root_dir)
    print(f"Found {len(files)} files to evaluate in {args.root_dir}.")
    
    global_id = 0
    
    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Skipping {file_path} due to read error: {e}")
            continue
            
        chunks = chunk_text(content, file_path.suffix)
        if not chunks:
            continue
            
        print(f"Processing {file_path.relative_to(Path(args.root_dir).resolve())} ({len(chunks)} chunks)...")
        
        # We batch our operations instead of inserting chunks one-by-one. 
        # This reduces round-trip overhead to the embedding server and the internal SQLite DB.
        batch_ids = []
        batch_documents = []
        batch_metadatas = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{file_path}_{i}"
            batch_ids.append(chunk_id)
            
            # Embed the filename context directly into the physical chunk
            augmented_chunk = f"File: {file_path.name}\nPath: {file_path}\n\n{chunk}"
            batch_documents.append(augmented_chunk)
            
            batch_metadatas.append({"source": str(file_path), "chunk_index": i})
            global_id += 1
                
        if batch_ids:
            collection.add(
                ids=batch_ids,
                documents=batch_documents,
                metadatas=batch_metadatas
            )

    print(f"\\nSuccess! Indexed {global_id} total chunks into the '{args.collection}' collection at {args.db_dir}.")

if __name__ == "__main__":
    main()

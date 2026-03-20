from random import Random
import typer
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from .client import AIClient
from .dbClient import DBClient
import os
import json
import time
import uuid
import chromadb
import ollama


app = typer.Typer(help="Ai_Oracle: Simple CLI AI assistant", no_args_is_help=True)
console = Console()
ai = AIClient()

def ask_internal(
    prompt: str = typer.Argument(..., help="A kérdésed az AI-hoz"),
    context: list | None = typer.Argument(None, help="Context"),
    model: str = typer.Option("llama3.2", "--model", "-m", help="Használt modell")
):
    ai.model = model
    full_response = ""
    safe_context = context if context is not None else []

    with Live(Markdown(""), console=console, refresh_per_second=10) as live:
        for chunk in ai.stream_chat(prompt, safe_context):
            full_response += chunk
            live.update(Markdown(full_response))

    return full_response        

@app.callback()
def main_callback():
    pass

@app.command()
def ask(
    prompt: str = typer.Argument(..., help="Ask with context"),
    userId: str = typer.Option("", "--user-id", "-u", help="User ID"),
    model: str = typer.Option("llama3.2", "--model", "-m", help="Used model")
):
    """Ask something"""
    with DBClient() as db:
        context = db.get_context(userId)
    
        full_response = ask_internal(prompt, context, model)
    
        db.add_context(userId, "user", prompt)
        db.add_context(userId, "assistant", full_response)
    
    return full_response

@app.command()
def clear_context(userId: str = typer.Option("", "--user-id", "-u", help="User ID")):
    """Clear context"""
    with DBClient() as db:
        db.clear_context(userId)
        console.print(f"Context cleared for user {userId}")

@app.command()
def list_context(userId: str = typer.Option("", "--user-id", "-u", help="User ID")):
    """List context"""
    with DBClient() as db:
        context = db.get_context(userId)
        if context:
            console.print(context)
        else:
            console.print("No context found for user " + userId)

@app.command()
def long_talk():
    """This is for longer talks"""
    userId = str(uuid.uuid4())
    while True:
        prompt = console.input("\n[bold blue]Question:[/bold blue] ")
        if prompt == "exit":
            break
        elif prompt == "clear-context":
            clear_context(userId)
        elif prompt == "list-context":
            list_context(userId)
        else:   
            ask(prompt, userId, "llama3.2")
    

@app.command()
def code_talk(
    prompt: str = typer.Argument(..., help="Ask questions about your codebase"),
    userId: str = typer.Option("", "--user-id", "-u", help="User ID"),
    model: str = typer.Option("llama3.2", "--model", "-m", help="Used model"),
    db_dir: str = typer.Option(os.path.expanduser("~/.ai_oracle/chroma_data"), "--db-dir", help="Chroma DB path"),
    collection_name: str = typer.Option("documents", "--collection", help="Chroma collection"),
    embed_model: str = typer.Option("nomic-embed-text", "--embed", help="Embedding model")
):
    """Talk based on code base retrieving data from ChromaDB. 
    It executes a semantic mathematical search, mixed with a structural substring search."""
    try:
        # Load the Chroma collection with our explicit embedding model. 
        # By providing `embedding_function`, Chroma handles vectorizing the strings automatically.
        from chromadb.utils import embedding_functions
        ollama_ef = embedding_functions.OllamaEmbeddingFunction(
            url="http://localhost:11434/api/embeddings",
            model_name=embed_model,
        )
        chroma_client = chromadb.PersistentClient(path=db_dir)
        collection = chroma_client.get_collection(name=collection_name, embedding_function=ollama_ef)
    except Exception as e:
        console.print(f"[bold red]Error connecting to ChromaDB at '{db_dir}':[/bold red] {e}")
        return
        
    import re
    
    # RAG Vector Search handles natural language (e.g. "What parses text data?"), but it handles 
    # exact terms poorly (e.g. "FileProcessingService.java" might fail against pure math distances).
    # We supplement standard text similarity by extracting CamelCase class names or .java file names...
    keywords = set(re.findall(r'\b[A-Z][a-zA-Z0-9_]+\b|\b[a-zA-Z0-9_]+\.[a-zA-Z0-9]+\b', prompt))
    
    # ... and discarding common english words that triggered the CamelCase Regex.
    stop_words = {"How", "What", "Why", "When", "Where", "Which", "Who", "Can", "Could", "Would", "Should", "Is", "Are", "Do", "Does", "Did", "The", "A", "An", "Please", "Show", "Tell"}
    keywords = [kw for kw in keywords if kw not in stop_words]

    retrieved_documents = []
    retrieved_metadatas = []

    # 1. Semantic Embedding Search
    # "Which file queries the database?"
    results = collection.query(
        query_texts=[prompt],
        n_results=15
    )
    if results['documents'] and results['documents'][0]:
        retrieved_documents.extend(results['documents'][0])
        retrieved_metadatas.extend(results['metadatas'][0])

    # 2. Strict Keyword Matching
    # "Does PostgresRepository.java exist in the codebase?"
    for kw in keywords:
        try:
            # We filter chunks physically containing the string kw (like 'PostgresRepository') natively at the SQLite level
            kw_results = collection.get(
                where_document={"$contains": kw},
                limit=10
            )
            if kw_results['documents']:
                retrieved_documents.extend(kw_results['documents'])
                retrieved_metadatas.extend(kw_results['metadatas'])
        except Exception as e:
            pass

    retrieved_context: str = ""
    seen_chunks = set()
    
    # Combine the semantic chunks and the string-matched chunks together into the prompt.
    for doc, meta in zip(retrieved_documents, retrieved_metadatas):
        source = meta.get('source', 'Unknown')
        chunk_index = meta.get('chunk_index', 0)
        
        # De-duplicate chunks since dense-search and string-search might retrieve the exact same block.
        chunk_id_key = f"{source}_{chunk_index}"
        if chunk_id_key not in seen_chunks:
            seen_chunks.add(chunk_id_key)
            # Add a clear separator marking where the code block came from for the LLM.
            retrieved_context += f"\n--- Source: {source} ---\n{doc}\n"
            
    if not retrieved_context:
        console.print("[yellow]No relevant context found in the vector database.[/yellow]")
        return
        

    # Compose the final context-heavy prompt using the user's original chat inputs plus our 
    # freshly curated VectorDB code slices so it's impossible for the LLM to hallucinate.
    augmented_prompt = (
        f"Answer the user's question based on the following code context.\n\n"
        f"CONTEXT:\n{retrieved_context}\n\n"
        f"QUESTION: {prompt}"
    )

    with DBClient() as db:
        chat_context = db.get_context(userId)
        
        full_response = ask_internal(augmented_prompt, chat_context, model)
    
        db.add_context(userId, "user", prompt)
        db.add_context(userId, "assistant", full_response)

    return full_response


if __name__ == "__main__":
    app()
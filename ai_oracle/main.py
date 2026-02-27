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


app = typer.Typer(help="Ai_Oracle: Simple CLI AI assistant", no_args_is_help=True)
console = Console()
ai = AIClient()

def ask_internal(
    prompt: str = typer.Argument(..., help="A kérdésed az AI-hoz"),
    context: list = typer.Argument([], help="Context"),
    model: str = typer.Option("llama3.2", "--model", "-m", help="Használt modell")
):
    ai.model = model
    full_response = ""

    with Live(Markdown(""), console=console, refresh_per_second=10) as live:
        for chunk in ai.stream_chat(prompt, context):
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
    


if __name__ == "__main__":
    app()
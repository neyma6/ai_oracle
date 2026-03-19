import os
import smtplib
from email.message import EmailMessage
from rich.console import Console

console = Console()

class EmailClient:
    def __init__(self):
        self.smtp_server = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.environ.get("SMTP_PORT", 587))
        self.smtp_user, self.smtp_password = self._load_credentials()

    def _load_credentials(self):
        email_creds_path = os.path.expanduser("~/.ai_oracle/email")
        if os.path.exists(email_creds_path):
            with open(email_creds_path, 'r') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
                if len(lines) >= 2:
                    return lines[0], lines[1]
        return None, None

    def send_email(self, to_email: str, subject: str, body: str, image_path: str):
        """Send an email with the analysis result and the captured image."""
        if not all([self.smtp_user, self.smtp_password]):
            email_creds_path = os.path.expanduser("~/.ai_oracle/email")
            console.print(f"[yellow]Warning:[/yellow] Credentials not found. Please create [bold]{email_creds_path}[/bold] with your email on the first line and password on the second. Cannot send email.")
            return

        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = self.smtp_user
        msg["To"] = to_email
        msg.set_content(body)

        # Attach the image
        if os.path.exists(image_path):
            import mimetypes
            with open(image_path, 'rb') as fp:
                img_data = fp.read()
            mime_type, _ = mimetypes.guess_type(image_path)
            subtype = mime_type.split('/')[1] if mime_type and '/' in mime_type else 'jpeg'
            msg.add_attachment(img_data, maintype='image', subtype=subtype, filename=os.path.basename(image_path))

        try:
            console.print(f"[cyan]Sending email to {to_email}...[/cyan]")
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)
            console.print("[green]Email sent successfully![/green]")
        except Exception as e:
            console.print(f"[bold red]Failed to send email:[/bold red] {e}")

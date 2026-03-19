import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class DBClient:
    def __init__(self, host="localhost", database="ai_oracle", user="postgres", password="postgres", port=5432):
        """Initialize the database client."""
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.port = port
        self.connection = None

    def connect(self):
        """Establish a connection to the PostgreSQL database."""
        if self.connection is None or self.connection.closed != 0:
            try:
                self.connection = psycopg2.connect(
                    host=self.host,
                    database=self.database,
                    user=self.user,
                    password=self.password,
                    port=self.port
                )
            except psycopg2.Error as e:
                logger.error(f"Error connecting to PostgreSQL database: {e}")
                raise

    def close(self):
        """Close the database connection."""
        if self.connection and self.connection.closed == 0:
            self.connection.close()

    def get_context(self, user_id):
        """Retrieve previous context for a user."""
        self.connect()
        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    "SELECT role, content FROM contexts WHERE user_id = %s ORDER BY created_at ASC",
                    (user_id,)
                )
                return [{"role": row["role"], "content": row["content"]} for row in cursor.fetchall()]
        except psycopg2.Error as e:
            if self.connection:
                self.connection.rollback()
            logger.error(f"Error fetching context: {e}")
            return []

    def add_context(self, user_id, role, content):
        """Store a new context entry for a user."""
        self.connect()
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO contexts (user_id, role, content) VALUES (%s, %s, %s)",
                    (user_id, role, content)
                )
                self.connection.commit()
        except psycopg2.Error as e:
            if self.connection:
                self.connection.rollback()
            logger.error(f"Error saving context: {e}")
            raise

    def clear_context(self, user_id):
        """Delete all context for a given user."""
        self.connect()
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("DELETE FROM contexts WHERE user_id = %s", (user_id,))
                self.connection.commit()
        except psycopg2.Error as e:
            if self.connection:
                self.connection.rollback()
            logger.error(f"Error clearing context: {e}")
            raise

    # ── Event Log persistence ──────────────────────────────────────────

    def save_event_log(self, time_str, classification, confidence, image_data=None):
        """Persist a single classification event to the database and return its id."""
        self.connect()
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO event_logs (time_str, classification, confidence, image_data) VALUES (%s, %s, %s, %s) RETURNING id",
                    (time_str, classification, confidence, image_data)
                )
                log_id = cursor.fetchone()[0]
                self.connection.commit()
                return log_id
        except psycopg2.Error as e:
            if self.connection:
                self.connection.rollback()
            logger.error(f"Error saving event log: {e}")

    def load_event_logs(self):
        """Load event logs from today and yesterday."""
        self.connect()
        try:
            yesterday = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    "SELECT id, time_str, classification, confidence, created_at "
                    "FROM event_logs WHERE created_at >= %s ORDER BY created_at ASC",
                    (yesterday,)
                )
                return cursor.fetchall()
        except psycopg2.Error as e:
            if self.connection:
                self.connection.rollback()
            logger.error(f"Error loading event logs: {e}")
            return []

    def get_event_image(self, log_id):
        """Load image data for a specific event log."""
        self.connect()
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT image_data FROM event_logs WHERE id = %s", (log_id,))
                row = cursor.fetchone()
                return bytes(row[0]) if row and row[0] is not None else None
        except psycopg2.Error as e:
            if self.connection:
                self.connection.rollback()
            logger.error(f"Error loading event image: {e}")
            return None

    # ── AI Analysis persistence ────────────────────────────────────────

    def save_ai_analysis(self, time_str, result_text, image_data=None):
        """Persist an AI analysis result to the database and return its id."""
        self.connect()
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO ai_analyses (time_str, result_text, image_data) VALUES (%s, %s, %s) RETURNING id",
                    (time_str, result_text, image_data)
                )
                log_id = cursor.fetchone()[0]
                self.connection.commit()
                return log_id
        except psycopg2.Error as e:
            if self.connection:
                self.connection.rollback()
            logger.error(f"Error saving AI analysis: {e}")

    def load_ai_analyses(self):
        """Load AI analysis results from today and yesterday."""
        self.connect()
        try:
            yesterday = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    "SELECT id, time_str, result_text, created_at "
                    "FROM ai_analyses WHERE created_at >= %s ORDER BY created_at ASC",
                    (yesterday,)
                )
                return cursor.fetchall()
        except psycopg2.Error as e:
            if self.connection:
                self.connection.rollback()
            logger.error(f"Error loading AI analyses: {e}")
            return []

    def get_ai_image(self, log_id):
        """Load image data for a specific AI analysis."""
        self.connect()
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT image_data FROM ai_analyses WHERE id = %s", (log_id,))
                row = cursor.fetchone()
                return bytes(row[0]) if row and row[0] is not None else None
        except psycopg2.Error as e:
            if self.connection:
                self.connection.rollback()
            logger.error(f"Error loading AI image: {e}")
            return None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

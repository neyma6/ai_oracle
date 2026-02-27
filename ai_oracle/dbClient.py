import psycopg2
from psycopg2.extras import RealDictCursor
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

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

from yoyo import step

steps = [
    step(
        "CREATE TABLE IF NOT EXISTS contexts ("
        "id SERIAL PRIMARY KEY, "
        "user_id VARCHAR(255) NOT NULL, "
        "role VARCHAR(50) NOT NULL, "
        "content TEXT NOT NULL, "
        "created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP)"
    )
]

from yoyo import step

steps = [
    step(
        "CREATE TABLE IF NOT EXISTS event_logs ("
        "id SERIAL PRIMARY KEY, "
        "time_str VARCHAR(20) NOT NULL, "
        "classification VARCHAR(255) NOT NULL, "
        "confidence VARCHAR(20) NOT NULL, "
        "created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP)"
    ),
    step(
        "CREATE TABLE IF NOT EXISTS ai_analyses ("
        "id SERIAL PRIMARY KEY, "
        "time_str VARCHAR(20) NOT NULL, "
        "result_text TEXT NOT NULL, "
        "created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP)"
    ),
]

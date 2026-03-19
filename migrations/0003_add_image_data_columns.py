from yoyo import step

steps = [
    step(
        "ALTER TABLE event_logs ADD COLUMN image_data BYTEA",
        "ALTER TABLE event_logs DROP COLUMN image_data"
    ),
    step(
        "ALTER TABLE ai_analyses ADD COLUMN image_data BYTEA",
        "ALTER TABLE ai_analyses DROP COLUMN image_data"
    )
]

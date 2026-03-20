from yoyo import step

steps = [
    step(
        "ALTER TABLE event_logs ADD COLUMN event_date DATE",
        "ALTER TABLE event_logs DROP COLUMN event_date"
    ),
    step(
        "ALTER TABLE ai_analyses ADD COLUMN event_date DATE",
        "ALTER TABLE ai_analyses DROP COLUMN event_date"
    )
]

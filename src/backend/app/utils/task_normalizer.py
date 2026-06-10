from app.schemas.task import TaskPayload


def normalize_task_payload(task: TaskPayload) -> None:
    """Normalize task payload fields in-place."""
    task.name = str(task.name).strip()
    if task.name:
        task.name = task.name[0].upper() + task.name[1:].lower()

    task.location = str(task.location).lower().strip() if task.location else "home"

    if task.category:
        task.category = [str(c).lower().strip() for c in task.category if c]
    else:
        task.category = []

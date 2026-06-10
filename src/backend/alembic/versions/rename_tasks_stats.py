"""rename tasks_statistics to task_statistics

Revision ID: rename_tasks_stats
Revises: 2667094139c9
Create Date: 2026-04-19 20:00:00.000000

"""

from typing import Sequence, Union
from alembic import op

revision: str = "rename_tasks_stats"
down_revision: Union[str, Sequence[str], None] = "2667094139c9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Rename tasks_statistics to task_statistics."""
    # Rename the table
    op.execute("ALTER TABLE tasks_statistics RENAME TO task_statistics")

    # Update foreign key constraints
    op.execute(
        "ALTER TABLE task_statistics_locations "
        "DROP CONSTRAINT IF EXISTS task_statistics_locations_statistics_id_fkey"
    )
    op.execute(
        "ALTER TABLE task_statistics_locations "
        "ADD CONSTRAINT task_statistics_locations_statistics_id_fkey "
        "FOREIGN KEY (statistics_id) REFERENCES task_statistics(id) ON DELETE CASCADE"
    )

    op.execute(
        "ALTER TABLE tasks DROP CONSTRAINT IF EXISTS tasks_task_statistics_id_fkey"
    )
    op.execute(
        "ALTER TABLE tasks "
        "ADD CONSTRAINT tasks_task_statistics_id_fkey "
        "FOREIGN KEY (task_statistics_id) REFERENCES task_statistics(id) ON DELETE NO ACTION"
    )

    op.execute(
        "ALTER TABLE tasks "
        "DROP CONSTRAINT IF EXISTS tasks_associated_task_statistics_id_fkey"
    )
    op.execute(
        "ALTER TABLE tasks "
        "ADD CONSTRAINT tasks_associated_task_statistics_id_fkey "
        "FOREIGN KEY (associated_task_statistics_id) REFERENCES task_statistics(id) ON DELETE NO ACTION"
    )


def downgrade() -> None:
    """Rename task_statistics back to tasks_statistics."""
    # Drop FK constraints
    op.execute(
        "ALTER TABLE task_statistics_locations "
        "DROP CONSTRAINT IF EXISTS task_statistics_locations_statistics_id_fkey"
    )
    op.execute(
        "ALTER TABLE tasks DROP CONSTRAINT IF EXISTS tasks_task_statistics_id_fkey"
    )
    op.execute(
        "ALTER TABLE tasks "
        "DROP CONSTRAINT IF EXISTS tasks_associated_task_statistics_id_fkey"
    )

    # Rename table back
    op.execute("ALTER TABLE task_statistics RENAME TO tasks_statistics")

    # Recreate FK constraints
    op.execute(
        "ALTER TABLE task_statistics_locations "
        "ADD CONSTRAINT task_statistics_locations_statistics_id_fkey "
        "FOREIGN KEY (statistics_id) REFERENCES tasks_statistics(id) ON DELETE CASCADE"
    )
    op.execute(
        "ALTER TABLE tasks "
        "ADD CONSTRAINT tasks_task_statistics_id_fkey "
        "FOREIGN KEY (task_statistics_id) REFERENCES tasks_statistics(id) ON DELETE NO ACTION"
    )
    op.execute(
        "ALTER TABLE tasks "
        "ADD CONSTRAINT tasks_associated_task_statistics_id_fkey "
        "FOREIGN KEY (associated_task_statistics_id) REFERENCES tasks_statistics(id) ON DELETE NO ACTION"
    )

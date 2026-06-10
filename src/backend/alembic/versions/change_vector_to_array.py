"""change task_name_vector to array type

Revision ID: change_vector_to_array
Revises: rename_tasks_stats
Create Date: 2026-04-19 21:30:00.000000

"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy.dialects.postgresql as postgresql

revision: str = "change_vector_to_array"
down_revision: Union[str, Sequence[str], None] = "rename_tasks_stats"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Change task_name_vector from FLOAT to ARRAY(FLOAT) using temp column."""
    # Since all values are NULL, use temp column approach
    op.execute(
        "ALTER TABLE task_statistics ADD COLUMN task_name_vector_new DOUBLE PRECISION[]"
    )
    op.execute("ALTER TABLE task_statistics DROP COLUMN task_name_vector")
    op.execute(
        "ALTER TABLE task_statistics RENAME COLUMN task_name_vector_new TO task_name_vector"
    )


def downgrade() -> None:
    """Change task_name_vector back to FLOAT."""
    op.execute(
        "ALTER TABLE task_statistics ALTER COLUMN task_name_vector TYPE DOUBLE PRECISION"
    )

"""add category_id to category_statistics

Revision ID: 2667094139c9
Revises: b1deeeceae45
Create Date: 2026-04-19 13:18:57.814595

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '2667094139c9'
down_revision: Union[str, Sequence[str], None] = 'b1deeeceae45'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Drop FK from junction table first
    op.drop_constraint('category_statistics_locations_statistics_id_fkey', 'category_statistics_locations', type_='foreignkey')
    
    # Create new table with UUID id
    op.create_table('category_statistics_new',
        sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('category_id', sa.UUID(), nullable=False),
        sa.Column('avg_duration', sa.JSON(), nullable=True),
        sa.Column('avg_duration_delta', sa.JSON(), nullable=True),
        sa.Column('avg_difficulty', sa.Float(), nullable=True),
        sa.Column('avg_difficulty_delta', sa.Float(), nullable=True),
        sa.Column('completed_count', sa.Integer(), nullable=False),
        sa.Column('uncompleted_count', sa.Integer(), nullable=False),
        sa.Column('records', sa.Integer(), nullable=False),
        sa.Column('category_time_scores', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    )
    
    # Copy data with new UUIDs
    op.execute("""
        INSERT INTO category_statistics_new (id, category_id, avg_duration, avg_duration_delta, 
            avg_difficulty, avg_difficulty_delta, completed_count, uncompleted_count, records, 
            category_time_scores, created_at, updated_at)
        SELECT gen_random_uuid(), c.id, cs.avg_duration, cs.avg_duration_delta,
            cs.avg_difficulty, cs.avg_difficulty_delta, cs.completed_count, cs.uncompleted_count,
            cs.records, cs.category_time_scores, cs.created_at, cs.updated_at
        FROM category_statistics cs
        JOIN categories c ON c.name = cs.category_name
    """)
    
    # Drop old table and rename new one
    op.drop_table('category_statistics')
    op.rename_table('category_statistics_new', 'category_statistics')
    
    # Add primary key and constraints
    op.create_primary_key('category_statistics_pkey', 'category_statistics', ['id'])
    op.create_unique_constraint('category_statistics_category_id_key', 'category_statistics', ['category_id'])
    op.create_foreign_key(None, 'category_statistics', 'categories', ['category_id'], ['id'], ondelete='CASCADE')
    
    # Rebuild junction table with UUID FK
    op.drop_table('category_statistics_locations')
    op.create_table('category_statistics_locations',
        sa.Column('statistics_id', sa.UUID(), nullable=False),
        sa.Column('location_id', sa.UUID(), nullable=False),
        sa.Column('count', sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(['location_id'], ['locations.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['statistics_id'], ['category_statistics.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('statistics_id', 'location_id')
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    """Downgrade schema."""
    # Drop junction table and recreate with INTEGER
    op.drop_table('category_statistics_locations')
    op.create_table('category_statistics_locations',
        sa.Column('statistics_id', sa.Integer(), nullable=False),
        sa.Column('location_id', sa.UUID(), nullable=False),
        sa.Column('count', sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(['location_id'], ['locations.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['statistics_id'], ['category_statistics.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('statistics_id', 'location_id')
    )
    
    # Drop FK and constraints
    op.drop_constraint(None, 'category_statistics', type_='foreignkey')
    op.drop_constraint('category_statistics_category_id_key', 'category_statistics', type_='unique')
    op.drop_primary_key('category_statistics_pkey', 'category_statistics')
    
    # Recreate old table with INTEGER id
    op.create_table('category_statistics_old',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('category_name', sa.Text(), nullable=False),
        sa.Column('avg_duration', sa.JSON(), nullable=True),
        sa.Column('avg_duration_delta', sa.JSON(), nullable=True),
        sa.Column('avg_difficulty', sa.Float(), nullable=True),
        sa.Column('avg_difficulty_delta', sa.Float(), nullable=True),
        sa.Column('completed_count', sa.Integer(), nullable=False),
        sa.Column('uncompleted_count', sa.Integer(), nullable=False),
        sa.Column('records', sa.Integer(), nullable=False),
        sa.Column('category_time_scores', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    )
    
    # Copy data back
    op.execute("""
        INSERT INTO category_statistics_old (id, category_name, avg_duration, avg_duration_delta,
            avg_difficulty, avg_difficulty_delta, completed_count, uncompleted_count, records,
            category_time_scores, created_at, updated_at)
        SELECT row_number() OVER(), c.name, cs.avg_duration, cs.avg_duration_delta,
            cs.avg_difficulty, cs.avg_difficulty_delta, cs.completed_count, cs.uncompleted_count,
            cs.records, cs.category_time_scores, cs.created_at, cs.updated_at
        FROM category_statistics cs
        JOIN categories c ON c.id = cs.category_id
    """)
    
    op.drop_table('category_statistics')
    op.rename_table('category_statistics_old', 'category_statistics')
    
    op.create_primary_key('category_statistics_pkey', 'category_statistics', ['id'])
    op.create_unique_constraint(op.f('category_statistics_category_name_key'), 'category_statistics', ['category_name'])
    # ### end Alembic commands ###

"""add adaptation tables

Revision ID: 1a2b3c4d5e6f
Revises: 0e87a4871404
Create Date: 2026-10-12 13:57:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '1a2b3c4d5e6f'
down_revision: Union[str, Sequence[str], None] = '0e87a4871404'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    from sqlalchemy.dialects import postgresql
    
    # Create enums if they don't exist
    conn = op.get_bind()
    
    # Check and create adaptationlevel
    result = conn.execute(sa.text(
        "SELECT 1 FROM pg_type WHERE typname = 'adaptationlevel'"
    )).fetchone()
    if not result:
        conn.execute(sa.text(
            "CREATE TYPE adaptationlevel AS ENUM ('rag', 'soft_prompt', 'lora')"
        ))
    
    # Check and create adaptationstatus
    result = conn.execute(sa.text(
        "SELECT 1 FROM pg_type WHERE typname = 'adaptationstatus'"
    )).fetchone()
    if not result:
        conn.execute(sa.text(
            "CREATE TYPE adaptationstatus AS ENUM ('pending', 'training', 'ready', 'failed')"
        ))
    
    # Reference the enums for columns
    adaptation_level = postgresql.ENUM('rag', 'soft_prompt', 'lora', name='adaptationlevel', create_type=False)
    adaptation_status = postgresql.ENUM('pending', 'training', 'ready', 'failed', name='adaptationstatus', create_type=False)
    
    # user_adaptations table
    op.create_table('user_adaptations',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('base_model_id', sa.Integer(), nullable=False),
        sa.Column('level', adaptation_level, nullable=False),
        sa.Column('status', adaptation_status, nullable=True),
        sa.Column('artifact_path', sa.String(500), nullable=True),
        sa.Column('config', postgresql.JSONB(), nullable=True),
        sa.Column('metrics', postgresql.JSONB(), nullable=True),
        sa.Column('contribution_count', sa.Integer(), default=0),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('last_trained_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['base_model_id'], ['models.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_user_adaptations_user_id'), 'user_adaptations', ['user_id'], unique=False)
    op.create_index(op.f('ix_user_adaptations_base_model_id'), 'user_adaptations', ['base_model_id'], unique=False)
    
    # image_embeddings table
    op.create_table('image_embeddings',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('image_id', sa.Integer(), nullable=False),
        sa.Column('model_name', sa.String(100), nullable=False),
        sa.Column('embedding', postgresql.JSONB(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['image_id'], ['images.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('image_id', 'model_name', name='uq_image_embedding_model')
    )
    op.create_index(op.f('ix_image_embeddings_image_id'), 'image_embeddings', ['image_id'], unique=False)
    
    # comparison_feedback table
    op.create_table('comparison_feedback',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('image_id', sa.Integer(), nullable=False),
        sa.Column('adaptation_id', sa.Integer(), nullable=False),
        sa.Column('prompt', sa.Text(), nullable=False),
        sa.Column('base_response', sa.Text(), nullable=False),
        sa.Column('adapted_response', sa.Text(), nullable=False),
        sa.Column('preference', sa.String(20), nullable=True),
        sa.Column('base_thumbs', sa.String(10), nullable=True),
        sa.Column('adapted_thumbs', sa.String(10), nullable=True),
        sa.Column('base_edit', sa.Text(), nullable=True),
        sa.Column('adapted_edit', sa.Text(), nullable=True),
        sa.Column('comment', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['image_id'], ['images.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['adaptation_id'], ['user_adaptations.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_comparison_feedback_user_id'), 'comparison_feedback', ['user_id'], unique=False)
    op.create_index(op.f('ix_comparison_feedback_image_id'), 'comparison_feedback', ['image_id'], unique=False)
    op.create_index(op.f('ix_comparison_feedback_adaptation_id'), 'comparison_feedback', ['adaptation_id'], unique=False)
    op.create_index(op.f('ix_comparison_feedback_created_at'), 'comparison_feedback', ['created_at'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_comparison_feedback_created_at'), table_name='comparison_feedback')
    op.drop_index(op.f('ix_comparison_feedback_adaptation_id'), table_name='comparison_feedback')
    op.drop_index(op.f('ix_comparison_feedback_image_id'), table_name='comparison_feedback')
    op.drop_index(op.f('ix_comparison_feedback_user_id'), table_name='comparison_feedback')
    op.drop_table('comparison_feedback')
    
    op.drop_index(op.f('ix_image_embeddings_image_id'), table_name='image_embeddings')
    op.drop_table('image_embeddings')
    
    op.drop_index(op.f('ix_user_adaptations_base_model_id'), table_name='user_adaptations')
    op.drop_index(op.f('ix_user_adaptations_user_id'), table_name='user_adaptations')
    op.drop_table('user_adaptations')
    

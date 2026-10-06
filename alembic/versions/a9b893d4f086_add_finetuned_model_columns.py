"""add_finetuned_model_columns

Revision ID: a9b893d4f086
Revises: 6eaa5af0c5b8
Create Date: 2026-03-04 18:49:26.936429

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a9b893d4f086'
down_revision: Union[str, Sequence[str], None] = '6eaa5af0c5b8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    # Add columns to models table
    op.add_column('models', sa.Column('model_path', sa.String(500), nullable=True))
    op.add_column('models', sa.Column('is_finetuned', sa.Boolean(), server_default='false', nullable=False))
    op.add_column('models', sa.Column('base_model', sa.String(200), nullable=True))


def downgrade():
    op.drop_column('models', 'base_model')
    op.drop_column('models', 'is_finetuned')
    op.drop_column('models', 'model_path')

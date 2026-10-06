"""add is_public to images

Revision ID: 019de26e195b
Revises: 07a8021860ec
Create Date: 2026-02-11 18:19:54.014913

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '019de26e195b'
down_revision: Union[str, Sequence[str], None] = '07a8021860ec'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None




def upgrade():
    # Step 1: add column nullable first
    op.add_column(
        "images",
        sa.Column("is_public", sa.Boolean(), nullable=True),
    )

    # Step 2: backfill existing rows
    op.execute("UPDATE images SET is_public = FALSE WHERE is_public IS NULL")

    # Step 3: set NOT NULL
    op.alter_column(
        "images",
        "is_public",
        nullable=False,
    )

    # Step 4: add server default
    op.alter_column(
        "images",
        "is_public",
        server_default=sa.text("false"),
    )


def downgrade():
    op.drop_column("images", "is_public")





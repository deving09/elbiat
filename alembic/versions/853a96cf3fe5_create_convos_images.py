"""create convos images

Revision ID: 853a96cf3fe5
Revises: bb1f36bba702
Create Date: 2026-01-30 00:26:17.449701

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '853a96cf3fe5'
down_revision: Union[str, Sequence[str], None] = 'bb1f36bba702'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # 1) add nullable
    op.add_column("images", sa.Column("user_id", sa.Integer(), nullable=True))

    # 2) backfill existing rows (choose a real user id!)
    op.execute("UPDATE images SET user_id = 1 WHERE user_id IS NULL")

    # 3) enforce NOT NULL
    op.alter_column("images", "user_id", nullable=False)

    # 4) add FK + index (optional but recommended)
    op.create_foreign_key(
        "fk_images_user_id_users",
        "images",
        "users",
        ["user_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_index("ix_images_user_id", "images", ["user_id"])

def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_images_user_id", table_name="images")
    op.drop_constraint("fk_images_user_id_users", "images", type_="foreignkey")
    op.drop_column("images", "user_id")

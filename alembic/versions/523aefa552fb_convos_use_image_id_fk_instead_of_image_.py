"""convos use image_id fk instead of image filename

Revision ID: 523aefa552fb
Revises: c8e5a19397f8
Create Date: 2026-01-30 05:35:39.527022

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '523aefa552fb'
down_revision: Union[str, Sequence[str], None] = 'c8e5a19397f8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # 1) add new column (nullable for backfill)
    op.add_column("convos", sa.Column("image_id", sa.Integer(), nullable=True))
    op.create_index("ix_convos_image_id", "convos", ["image_id"])
    
    # 2) backfill: match convos.image to images.image_path exactly
    op.execute("""
        UPDATE convos c
        SET image_id = i.id
        FROM images i
        WHERE c.image_id IS NULL
          AND c.image = i.image_path
    """)

    # 3) backfill: match by basename (convos.image holds filename, images.image_path holds 'images/<fn>')
    op.execute("""
        UPDATE convos c
        SET image_id = i.id
        FROM images i
        WHERE c.image_id IS NULL
          AND c.image = regexp_replace(i.image_path, '^.*/', '')
    """)

    # 4) enforce NOT NULL after backfill
    # (this will fail if any rows couldn't be matched — which is good, you want to know)
    op.alter_column("convos", "image_id", nullable=False)

    # 5) add FK constraint
    op.create_foreign_key(
        "fk_convos_image_id_images",
        "convos",
        "images",
        ["image_id"],
        ["id"],
        ondelete="CASCADE",
    )

    # 6) drop old image column
    op.drop_column("convos", "image")



def downgrade() -> None:
    """Downgrade schema."""

    # add old column back (nullable because we can't reconstruct perfectly)
    op.add_column("convos", sa.Column("image", sa.String(), nullable=True))

    # best-effort restore from images.image_path
    op.execute("""
        UPDATE convos c
        SET image = i.image_path
        FROM images i
        WHERE c.image_id = i.id
    """)

    op.drop_constraint("fk_convos_image_id_images", "convos", type_="foreignkey")
    op.drop_index("ix_convos_image_id", table_name="convos")
    op.drop_column("convos", "image_id")

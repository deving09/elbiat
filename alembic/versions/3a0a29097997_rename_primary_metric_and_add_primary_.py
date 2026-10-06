"""rename primary_metric and add primary_metric_key

Revision ID: 3a0a29097997
Revises: 019de26e195b
Create Date: 2026-02-12 23:56:00.890523

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '3a0a29097997'
down_revision: Union[str, Sequence[str], None] = '019de26e195b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None



def upgrade():
    # 1️⃣ Rename column
    op.alter_column(
        "tasks",
        "primary_metric",
        new_column_name="primary_metric_type"
    )

    # 2️⃣ Add new column (nullable first)
    op.add_column(
        "tasks",
        sa.Column(
            "primary_metric_key",
            sa.String(),
            nullable=True
        )
    )

    # 3️⃣ Backfill existing rows
    op.execute(
        "UPDATE tasks SET primary_metric_key = 'avg' WHERE primary_metric_key IS NULL"
    )

    # 4️⃣ Make NOT NULL
    op.alter_column(
        "tasks",
        "primary_metric_key",
        nullable=False,
        server_default=sa.text("'avg'")
    )


def downgrade():
    op.drop_column("tasks", "primary_metric_key")

    op.alter_column(
        "tasks",
        "primary_metric_type",
        new_column_name="primary_metric"
    )

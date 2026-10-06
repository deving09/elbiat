"""Add role and experiment_bucket to users

Revision ID: 657f737954bf
Revises: 122d836226d7
Create Date: 2026-10-06 18:02:22.933814

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '657f737954bf'
down_revision: Union[str, Sequence[str], None] = '122d836226d7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None



def upgrade() -> None:
    # Create the enum type FIRST
    userrole = sa.Enum('user', 'enterprise', 'admin', name='userrole')
    userrole.create(op.get_bind(), checkfirst=True)
    
    # Now add columns
    op.add_column('users', sa.Column('role', userrole, server_default='user', nullable=False))
    op.add_column('users', sa.Column('experiment_bucket', sa.String(50), nullable=True))
    op.create_index('ix_users_experiment_bucket', 'users', ['experiment_bucket'])


def downgrade() -> None:
    op.drop_index('ix_users_experiment_bucket', 'users')
    op.drop_column('users', 'experiment_bucket')
    op.drop_column('users', 'role')
    
    # Drop the enum type
    sa.Enum(name='userrole').drop(op.get_bind(), checkfirst=True)

    # ### end Alembic commands ###

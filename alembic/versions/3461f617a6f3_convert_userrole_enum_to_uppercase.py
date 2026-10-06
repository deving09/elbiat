"""Convert userrole enum to uppercase

Revision ID: 3461f617a6f3
Revises: 657f737954bf
Create Date: 2026-10-06 19:40:56.303868

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '3461f617a6f3'
down_revision: Union[str, Sequence[str], None] = '657f737954bf'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None



def upgrade() -> None:
    # Drop the default first
    op.execute("ALTER TABLE users ALTER COLUMN role DROP DEFAULT")
    
    # Rename old enum
    op.execute("ALTER TYPE userrole RENAME TO userrole_old")
    
    # Create new enum with uppercase values
    op.execute("CREATE TYPE userrole AS ENUM ('USER', 'ENTERPRISE', 'ADMIN')")
    
    # Convert column to new enum
    op.execute("""
        ALTER TABLE users 
        ALTER COLUMN role TYPE userrole 
        USING UPPER(role::text)::userrole
    """)
    
    # Re-add default with uppercase
    op.execute("ALTER TABLE users ALTER COLUMN role SET DEFAULT 'USER'")
    
    # Drop old enum
    op.execute("DROP TYPE userrole_old")


def downgrade() -> None:
    op.execute("ALTER TABLE users ALTER COLUMN role DROP DEFAULT")
    op.execute("ALTER TYPE userrole RENAME TO userrole_old")
    op.execute("CREATE TYPE userrole AS ENUM ('user', 'enterprise', 'admin')")
    op.execute("""
        ALTER TABLE users 
        ALTER COLUMN role TYPE userrole 
        USING LOWER(role::text)::userrole
    """)
    op.execute("ALTER TABLE users ALTER COLUMN role SET DEFAULT 'user'")
    op.execute("DROP TYPE userrole_old")



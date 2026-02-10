""create evaluation tables

Revision ID: 001_create_eval_tables
Revises: <your_previous_revision>
Create Date: 2025-02-04

Creates tables for:
- tasks: evaluation benchmarks (CharXiv, ChartQA, etc.)
- models: ML models being evaluated
- eval_runs: individual evaluation runs
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001_create_eval_tables'
down_revision = None  # Set this to your previous migration revision
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create tasks table
    op.create_table(
        'tasks',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('display_name', sa.String(200), nullable=False),
        sa.Column('vlmeval_data', sa.String(100), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('primary_metric_suffix', sa.String(50), nullable=False, server_default='_acc.csv'),
        sa.Column('primary_metric_key', sa.String(50), nullable=False, server_default='acc'),
        sa.Column('dataset_version', sa.String(100), nullable=True),
        sa.Column('num_examples', sa.Integer(), nullable=True),
        sa.Column('paper_url', sa.String(500), nullable=True),
        sa.Column('dataset_url', sa.String(500), nullable=True),
        sa.Column('config', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_tasks_name', 'tasks', ['name'], unique=True)

    # Create models table
    op.create_table(
        'models',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('display_name', sa.String(200), nullable=False),
        sa.Column('vlmeval_model', sa.String(100), nullable=False),
        sa.Column('model_type', sa.String(10), nullable=False, server_default='vlm'),
        sa.Column('hf_id', sa.String(300), nullable=True),
        sa.Column('endpoint_url', sa.String(500), nullable=True),
        sa.Column('default_args', sa.JSON(), nullable=True),
        sa.Column('params_b', sa.Float(), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_models_name', 'models', ['name'], unique=True)

    # Create eval_runs table
    op.create_table(
        'eval_runs',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('task_id', sa.Integer(), nullable=False),
        sa.Column('model_id', sa.Integer(), nullable=False),
        sa.Column('status', sa.String(20), nullable=False, server_default='queued'),
        sa.Column('metrics', sa.JSON(), nullable=True),
        sa.Column('artifacts_dir', sa.String(500), nullable=True),
        sa.Column('command', sa.Text(), nullable=True),
        sa.Column('config_snapshot', sa.JSON(), nullable=True),
        sa.Column('git_commit', sa.String(40), nullable=True),
        sa.Column('error', sa.Text(), nullable=True),
        sa.Column('created_by_user_id', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['task_id'], ['tasks.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['model_id'], ['models.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['created_by_user_id'], ['users.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_eval_runs_task_id', 'eval_runs', ['task_id'])
    op.create_index('ix_eval_runs_model_id', 'eval_runs', ['model_id'])
    op.create_index('ix_eval_runs_status', 'eval_runs', ['status'])
    op.create_index('ix_eval_runs_task_model', 'eval_runs', ['task_id', 'model_id'])
    op.create_index('ix_eval_runs_status_created', 'eval_runs', ['status', 'created_at'])


def downgrade() -> None:
    op.drop_table('eval_runs')
    op.drop_table('models')
    op.drop_table('tasks')

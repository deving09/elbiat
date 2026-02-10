#!/usr/bin/env python3
"""
Seed script to populate initial tasks and models.

Run with: python seed_evals.py
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Database URL
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://postgres:postgres@localhost:5432/app"
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)


# =============================================================================
# Initial Tasks
# =============================================================================

"""{
        "name": "charxiv",
        "display_name": "CharXiv",
        "vlmeval_data": "CharXiv",  # Verify this is correct for VLMEvalKit
        "description": '''CharXiv is a comprehensive evaluation suite for chart understanding 
in Multimodal LLMs. It contains 2,323 natural, challenging, and diverse charts from 
scientific papers on arXiv. The benchmark includes descriptive questions (examining 
basic chart elements) and reasoning questions (synthesizing information across 
complex visual elements).''',
        "primary_metric_suffix": "_acc.csv",
        "primary_metric_key": "acc",
        "dataset_version": "v1.0",
        "num_examples": 2323,
        "paper_url": "https://arxiv.org/abs/2406.18521",
        "dataset_url": "https://huggingface.co/datasets/princeton-nlp/CharXiv",
    },"""


TASKS = [
   
    {
        "name": "chartqa_test",
        "display_name": "ChartQA TEST",
        "vlmeval_data": "ChartQA_TEST",
        "description": """ChartQA is a benchmark for question answering about charts with 
visual and logical reasoning. It contains 9.6K human-written questions and 23.1K 
machine-generated questions. Questions require both visual understanding and 
logical/arithmetic reasoning.""",
        "primary_metric_suffix": "_acc.csv",
        "primary_metric": "acc",
        "dataset_version": "test",
        "paper_url": "https://aclanthology.org/2022.findings-acl.177/",
        "dataset_url": "https://github.com/vis-nlp/ChartQA",
        "user_id": 1
    },
    {
        "name": "mme",
        "display_name": "MME",
        "vlmeval_data": "MME",
        "description": """MME is a comprehensive evaluation benchmark for Multimodal Large 
Language Models. It measures both perception and cognition abilities across 14 subtasks.""",
        "primary_metric_suffix": "_score.csv",
        "primary_metric": "score",
        "user_id": 1
    },
    {
        "name": "plotqa",
        "display_name": "PlotQA",
        "vlmeval_data": "PlotQA",
        "description": """PlotQA focuses on reasoning over scientific plots. It contains 
millions of question-answer pairs over bar charts, line graphs, and dot plots, 
requiring structural understanding and numerical reasoning.""",
        "primary_metric_suffix": "_acc.csv",
        "primary_metric": "acc",
        "paper_url": "https://arxiv.org/abs/1909.00997",
        "user_id": 1
    },
]


# =============================================================================
# Initial Models
# =============================================================================

MODELS = [
    {
        "name": "internvl2_5_2b",
        "display_name": "InternVL2.5 2B",
        "vlmeval_model": "InternVL2_5-2B",
        "model_type": "vlm",
        #"hf_id": "OpenGVLab/InternVL2_5-2B",
        #"params_b": 2.0,
        #"description": "InternVL 2.5 with 2 billion parameters",
    },
    {
        "name": "internvl2_5_8b",
        "display_name": "InternVL2.5 8B",
        "vlmeval_model": "InternVL2_5-8B",
        "model_type": "vlm",
        #"hf_id": "OpenGVLab/InternVL2_5-8B",
        #"params_b": 8.0,
        #"description": "InternVL 2.5 with 8 billion parameters",
    },
    {
        "name": "qwen2_vl_7b",
        "display_name": "Qwen2-VL 7B",
        "vlmeval_model": "Qwen2-VL-7B-Instruct",
        "model_type": "vlm",
        #"hf_id": "Qwen/Qwen2-VL-7B-Instruct",
        #"params_b": 7.0,
        #"description": "Qwen2-VL instruction-tuned 7B model",
    },
    {
        "name": "llava_onevision_7b",
        "display_name": "LLaVA OneVision 7B",
        "vlmeval_model": "LLaVA-OneVision-7B",
        "model_type": "vlm",
        #"params_b": 7.0,
    },
    {
        "name": "phi3_vision",
        "display_name": "Phi-3 Vision",
        "vlmeval_model": "Phi-3-Vision",
        "model_type": "vlm",
        #"hf_id": "microsoft/Phi-3-vision-128k-instruct",
        #"params_b": 4.2,
    },
]


def seed_tasks(session):
    """Seed tasks table."""
    from sqlalchemy import text
    
    for task_data in TASKS:
        # Check if exists
        result = session.execute(
            text("SELECT id FROM tasks WHERE name = :name"),
            {"name": task_data["name"]}
        )
        if result.fetchone():
            print(f"Task '{task_data['name']}' already exists, skipping")
            continue
        
        # Insert
        columns = ", ".join(task_data.keys())
        placeholders = ", ".join(f":{k}" for k in task_data.keys())
        session.execute(
            text(f"INSERT INTO tasks ({columns}) VALUES ({placeholders})"),
            task_data
        )
        print(f"Created task: {task_data['name']}")
    
    session.commit()


def seed_models(session):
    """Seed models table."""
    from sqlalchemy import text
    
    for model_data in MODELS:
        # Check if exists
        result = session.execute(
            text("SELECT id FROM models WHERE name = :name"),
            {"name": model_data["name"]}
        )
        if result.fetchone():
            print(f"Model '{model_data['name']}' already exists, skipping")
            continue
        
        # Insert
        columns = ", ".join(model_data.keys())
        placeholders = ", ".join(f":{k}" for k in model_data.keys())
        session.execute(
            text(f"INSERT INTO models ({columns}) VALUES ({placeholders})"),
            model_data
        )
        print(f"Created model: {model_data['name']}")
    
    session.commit()


def main():
    print("Seeding evaluation database...")
    print(f"Database: {DATABASE_URL.split('@')[-1]}")
    
    session = SessionLocal()
    try:
        seed_tasks(session)
        seed_models(session)
        print("\nSeeding complete!")
    finally:
        session.close()


if __name__ == "__main__":
    main()

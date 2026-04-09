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
"""
__tablename__ = "tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    name: Mapped[str] = mapped_column(String, unique=True, index=True)
    
    display_name: Mapped[str] = mapped_column(String)

    vlmeval_data: Mapped[str] = mapped_column(String, index=True)

    description: Mapped[str] = mapped_column(String)

    #primary_metric: Mapped[str] = mapped_column(String)

     # 🔁 Renamed
    primary_metric_type: Mapped[str] = mapped_column(String)

    # ➕ New field
    primary_metric_key: Mapped[str] = mapped_column(
        String,
        nullable=False,
        default="avg",                     # ORM default
        server_default="avg",              # DB default
    )


    primary_metric_suffix: Mapped[str] = mapped_column(String)

    num_examples: Mapped[int] = mapped_column(Integer, nullable=True)

    paper_url: Mapped[str] = mapped_column(String, nullable=True)

    dataset_url: Mapped[str] = mapped_column(String, nullable=True)

    dataset_version: Mapped[str] = mapped_column(String, nullable=True)

    user_id: Mapped[int] = mapped_column(Integer, 
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )
"""



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
        "primary_metric_type": "acc",
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
        "primary_metric_type": "score",
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
        "primary_metric_type": "acc",
        "paper_url": "https://arxiv.org/abs/1909.00997",
        "user_id": 1
    },

    {
        "name": "pope",
        "display_name": "POPE",
        "vlmeval_data": "POPE",
        "description": """POPE (Polling-based Object Probing Evaluation) is a benchmark 
for evaluating object hallucination in large vision-language models. It uses a 
polling-based query method with Yes/No questions to probe whether objects exist 
in images, measuring precision, recall, F1, and accuracy.""",
        "primary_metric_suffix": "_acc.csv",
        "primary_metric_type": "acc",
        "paper_url": "https://arxiv.org/abs/2305.10355",
        "dataset_url": "https://github.com/RUCAIBox/POPE",
        "user_id": 1
    },
    {
        "name": "mmstar",
        "display_name": "MMStar",
        "vlmeval_data": "MMStar",
        "description": """MMStar is an elite vision-indispensable multi-modal benchmark 
comprising 1,500 challenge samples meticulously selected by humans. It addresses two 
key issues in VLM evaluation: visual content being unnecessary for many samples, and 
unintentional data leakage in LLM/LVLM training. MMStar evaluates 6 core capabilities 
across 18 detailed axes.""",
        "primary_metric_suffix": "_acc.csv",
        "primary_metric_type": "acc",
        "paper_url": "https://arxiv.org/abs/2403.20330",
        "dataset_url": "https://github.com/MMStar-Benchmark/MMStar",
        "user_id": 1
    },
    {
        "name": "blink",
        "display_name": "BLINK",
        "vlmeval_data": "BLINK",
        "description": """BLINK is a benchmark designed to test visual perception abilities 
that multimodal LLMs struggle with. It contains tasks that are easy for humans but 
challenging for models, focusing on core visual perception skills like relative depth, 
visual correspondence, forensics detection, jigsaw puzzles, and multi-view reasoning.""",
        "primary_metric_suffix": "_acc.csv",
        "primary_metric_type": "acc",
        "paper_url": "https://arxiv.org/abs/2404.12390",
        "dataset_url": "https://zeyofu.github.io/blink/",
        "user_id": 1
    },
    {
        "name": "cvbench_2d",
        "display_name": "CV-Bench 2D",
        "vlmeval_data": "CV-Bench-2D",
        "description": """CV-Bench 2D evaluates 2D spatial understanding and reasoning 
capabilities. It tests abilities like object localization, spatial relationships 
(left/right, above/below), distance estimation, and size comparison in 2D image space.""",
        "primary_metric_suffix": "_acc.csv",
        "primary_metric_type": "acc",
        "paper_url": "https://arxiv.org/abs/2406.08290",
        "dataset_url": "https://github.com/cambrian-mllm/cambrian",
        "user_id": 1
    },
    {
        "name": "cvbench_3d",
        "display_name": "CV-Bench 3D",
        "vlmeval_data": "CV-Bench-3D",
        "description": """CV-Bench 3D evaluates 3D spatial understanding and depth 
reasoning capabilities. It tests abilities like depth ordering, 3D spatial 
relationships, distance estimation in 3D space, and understanding of camera 
perspective and viewpoint.""",
        "primary_metric_suffix": "_acc.csv",
        "primary_metric_type": "acc",
        "paper_url": "https://arxiv.org/abs/2406.08290",
        "dataset_url": "https://github.com/cambrian-mllm/cambrian",
        "user_id": 1
    },
    {
        "name": "mmvp",
        "display_name": "MMVP",
        "vlmeval_data": "MMVP",
        "description": """MMVP (Multimodal Visual Patterns) is a benchmark that identifies 
visual patterns where CLIP-based vision encoders struggle, leading to systematic failures 
in multimodal LLMs. It contains challenging visual perception pairs that test fine-grained 
visual understanding, orientation, counting, and visual properties that text encoders miss.""",
        "primary_metric_suffix": "_acc.csv",
        "primary_metric_type": "acc",
        "paper_url": "https://arxiv.org/abs/2401.06209",
        "dataset_url": "https://github.com/tsb0601/MMVP",
        "user_id": 1
    },

    {
        "name": "mochi_naive",
        "display_name": "MOCHI (Naive Multi-Image)",
        "vlmeval_data": "MOCHI_Naive",
        "description": """MOCHI (Multiview Object Consistency in Humans and Image models) 
tests 3D shape understanding through odd-one-out tasks. This naive approach passes 
multiple images (3-4) separately to the VLM, testing multi-image reasoning. Each 
trial shows views of objects where one is different from the others.""",
        "primary_metric_suffix": "_acc.csv",
        "primary_metric_type": "acc",

        "paper_url": "https://arxiv.org/abs/2409.05862",
        "dataset_url": "https://huggingface.co/datasets/tzler/MOCHI",
        "user_id": 1
    },
    {
        "name": "mochi_grid",
        "display_name": "MOCHI (Single Grid Image)",
        "vlmeval_data": "MOCHI_Grid",
        "description": """MOCHI (Multiview Object Consistency in Humans and Image models) 
tests 3D shape understanding through odd-one-out tasks. This grid approach merges 
multiple images into a labeled 2x2 grid (A, B, C, D), testing spatial layout 
understanding. Each trial shows views of objects where one is different.""",
        "primary_metric_suffix": "_acc.csv",
        "primary_metric_type": "acc",
        "paper_url": "https://arxiv.org/abs/2409.05862",
        "dataset_url": "https://huggingface.co/datasets/tzler/MOCHI",
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

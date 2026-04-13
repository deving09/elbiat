# Add these to your TASKS list in seed_evals.py

MOCHI_TASKS = [
    {
        "name": "mochi_naive",
        "display_name": "MOCHI (Naive Multi-Image)",
        "vlmeval_data": "MOCHI_Naive",
        "description": """MOCHI (Multiview Object Consistency in Humans and Image models) 
tests 3D shape understanding through odd-one-out tasks. This naive approach passes 
multiple images (3-4) separately to the VLM, testing multi-image reasoning. Each 
trial shows views of objects where one is different from the others.""",
        "primary_metric_suffix": "_acc.csv",
        "primary_metric": "acc",
        "task_type": "mcq",
        "category": "3d_understanding",
        "size": 2019,
        "requires_api_judge": False,
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
        "primary_metric": "acc",
        "task_type": "mcq",
        "category": "3d_understanding",
        "size": 2019,
        "requires_api_judge": False,
        "paper_url": "https://arxiv.org/abs/2409.05862",
        "dataset_url": "https://huggingface.co/datasets/tzler/MOCHI",
        "user_id": 1
    },
]

# Your full TASKS list should include these plus the previous ones:
"""
TASKS = [
    # Existing tasks...
    {
        "name": "chartqa_test",
        "display_name": "ChartQA TEST",
        "vlmeval_data": "ChartQA_TEST",
        ...
    },
    {
        "name": "mme",
        ...
    },
    # ... other existing tasks ...
    
    # New benchmarks
    {
        "name": "pope",
        "display_name": "POPE",
        "vlmeval_data": "POPE",
        ...
    },
    {
        "name": "mmstar",
        "display_name": "MMStar",
        "vlmeval_data": "MMStar",
        ...
    },
    # ... etc ...
    
    # MOCHI tasks
    *MOCHI_TASKS,  # Unpack the list
]
"""

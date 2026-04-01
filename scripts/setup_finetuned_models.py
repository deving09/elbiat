# scripts/setup_finetuned_models.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from app.db import DATABASE_URL

engine = create_engine(DATABASE_URL)

with engine.connect() as conn:
    # Add columns if not exist
    conn.execute(text("""
        ALTER TABLE models ADD COLUMN IF NOT EXISTS model_path VARCHAR(500);
    """))
    conn.execute(text("""
        ALTER TABLE models ADD COLUMN IF NOT EXISTS is_finetuned BOOLEAN DEFAULT FALSE;
    """))
    conn.execute(text("""
        ALTER TABLE models ADD COLUMN IF NOT EXISTS base_model VARCHAR(200);
    """))
    conn.commit()
    print("Added columns to models table")

    # Insert the finetuned model
    conn.execute(text("""
        INSERT INTO models (name, display_name, vlmeval_model, model_path, is_finetuned, base_model)
        VALUES (
            'internvl2_5_2b_refined_v1',
            'InternVL2.5-2B (Refined v1)',
            'InternVL2_5-2B-Refined-v1',
            '/home/ubuntu/workspace/elbiat/checkpoints/refined_v1_2b',
            true,
            'OpenGVLab/InternVL2_5-2B'
        )
        ON CONFLICT (name) DO UPDATE SET
            model_path = EXCLUDED.model_path,
            is_finetuned = EXCLUDED.is_finetuned,
            base_model = EXCLUDED.base_model;
    """))
    conn.commit()

    # Insert the finetuned model
    conn.execute(text("""
        INSERT INTO models (name, display_name, vlmeval_model, model_path, is_finetuned, base_model)
        VALUES (
            'internvl2_5_2b_refined_v2',
            'InternVL2.5-2B (Refined v2)',
            'InternVL2_5-2B-Refined-v2',
            '/home/ubuntu/workspace/elbiat/checkpoints/refined_v2_2b',
            true,
            'OpenGVLab/InternVL2_5-2B'
        )
        ON CONFLICT (name) DO UPDATE SET
            model_path = EXCLUDED.model_path,
            is_finetuned = EXCLUDED.is_finetuned,
            base_model = EXCLUDED.base_model;
    """))
    conn.commit()

    # Insert the finetuned model
    conn.execute(text("""
        INSERT INTO models (name, display_name, vlmeval_model, model_path, is_finetuned, base_model)
        VALUES (
            'internvl2_5_2b_feedback_v1',
            'InternVL2.5-2B (Feedback v1)',
            'InternVL2_5-2B-Feedback-v1',
            '/home/ubuntu/workspace/elbiat/checkpoints/feedback_v1_2b',
            true,
            'OpenGVLab/InternVL2_5-2B'
        )
        ON CONFLICT (name) DO UPDATE SET
            model_path = EXCLUDED.model_path,
            is_finetuned = EXCLUDED.is_finetuned,
            base_model = EXCLUDED.base_model;
    """))
    conn.commit()

    # Insert the finetuned model
    conn.execute(text("""
        INSERT INTO models (name, display_name, vlmeval_model, model_path, is_finetuned, base_model)
        VALUES (
            'internvl2_5_2b_feedback_v1_1ep',
            'InternVL2.5-2B (Feedback v1- 1ep)',
            'InternVL2_5-2B-Feedback-v1_1ep',
            '/home/ubuntu/workspace/elbiat/checkpoints/feedback_v1_2b_1e',
            true,
            'OpenGVLab/InternVL2_5-2B'
        )
        ON CONFLICT (name) DO UPDATE SET
            model_path = EXCLUDED.model_path,
            is_finetuned = EXCLUDED.is_finetuned,
            base_model = EXCLUDED.base_model;
    """))
    conn.commit()

    # Insert the finetuned model
    conn.execute(text("""
        INSERT INTO models (name, display_name, vlmeval_model, model_path, is_finetuned, base_model)
        VALUES (
            'internvl2_5_2b_feedback_v1_1000ep',
            'InternVL2.5-2B (Feedback v1- 1000ep)',
            'InternVL2_5-2B-Feedback-v1_1000ep',
            '/home/ubuntu/workspace/elbiat/checkpoints/feedback_v1_2b_1000e',
            true,
            'OpenGVLab/InternVL2_5-2B'
        )
        ON CONFLICT (name) DO UPDATE SET
            model_path = EXCLUDED.model_path,
            is_finetuned = EXCLUDED.is_finetuned,
            base_model = EXCLUDED.base_model;
    """))
    conn.commit()

        # Insert the finetuned model
    conn.execute(text("""
        INSERT INTO models (name, display_name, vlmeval_model, model_path, is_finetuned, base_model)
        VALUES (
            'internvl2_5_2b_full_v1_10ep',
            'InternVL2.5-2B (Full v1- 10ep)',
            'InternVL2_5-2B-Full-v1_10ep',
            '/home/ubuntu/workspace/elbiat/checkpoints/full_v1_2b_10e',
            true,
            'OpenGVLab/InternVL2_5-2B'
        )
        ON CONFLICT (name) DO UPDATE SET
            model_path = EXCLUDED.model_path,
            is_finetuned = EXCLUDED.is_finetuned,
            base_model = EXCLUDED.base_model;
    """))
    conn.commit()

    conn.execute(text("""
        INSERT INTO models (name, display_name, vlmeval_model, model_path, is_finetuned, base_model)
        VALUES (
            'internvl2_5_2b_full_v1_1ep',
            'InternVL2.5-2B (Full v1- 1ep)',
            'InternVL2_5-2B-Full-v1_1ep',
            '/home/ubuntu/workspace/elbiat/checkpoints/full_v1_2b_1e',
            true,
            'OpenGVLab/InternVL2_5-2B'
        )
        ON CONFLICT (name) DO UPDATE SET
            model_path = EXCLUDED.model_path,
            is_finetuned = EXCLUDED.is_finetuned,
            base_model = EXCLUDED.base_model;
    """))
    conn.commit()

    conn.execute(text("""
        INSERT INTO models (name, display_name, vlmeval_model, model_path, is_finetuned, base_model)
        VALUES (
            'internvl2_5_2b_feedback_llm_v1_1ep',
            'InternVL2.5-2B (Feedback LLM v1- 1ep)',
            'InternVL2_5-2B-Feedback-llm-v1_1ep',
            '/home/ubuntu/workspace/elbiat/checkpoints/full_feedback_llm_v1',
            true,
            'OpenGVLab/InternVL2_5-2B'
        )
        ON CONFLICT (name) DO UPDATE SET
            model_path = EXCLUDED.model_path,
            is_finetuned = EXCLUDED.is_finetuned,
            base_model = EXCLUDED.base_model;
    """))
    conn.commit()

    conn.execute(text("""
        INSERT INTO models (name, display_name, vlmeval_model, model_path, is_finetuned, base_model)
        VALUES (
            'internvl2_5_2b_feedback_llm_v1_10ep',
            'InternVL2.5-2B (Feedback LLM v1- 10ep)',
            'InternVL2_5-2B-Feedback-llm-v1_10ep',
            '/home/ubuntu/workspace/elbiat/checkpoints/full_feedback_llm_v1_10ep',
            true,
            'OpenGVLab/InternVL2_5-2B'
        )
        ON CONFLICT (name) DO UPDATE SET
            model_path = EXCLUDED.model_path,
            is_finetuned = EXCLUDED.is_finetuned,
            base_model = EXCLUDED.base_model;
    """))
    conn.commit()


    conn.execute(text("""
        INSERT INTO models (name, display_name, vlmeval_model, model_path, is_finetuned, base_model)
        VALUES (
            'internvl2_5_2b_feedback_vision_llm_v1_10ep',
            'InternVL2.5-2B (Feedback Vision LLM v1- 10ep)',
            'InternVL2_5-2B-Feedback-vision-llm-v1_10ep',
            '/home/ubuntu/workspace/elbiat/checkpoints/feedback_vision_llm_v1_10ep',
            true,
            'OpenGVLab/InternVL2_5-2B'
        )
        ON CONFLICT (name) DO UPDATE SET
            model_path = EXCLUDED.model_path,
            is_finetuned = EXCLUDED.is_finetuned,
            base_model = EXCLUDED.base_model;
    """))
    conn.commit()

    conn.execute(text("""
        INSERT INTO models (name, display_name, vlmeval_model, model_path, is_finetuned, base_model)
        VALUES (
            'internvl2_5_2b_feedback_vision_llm_v1_1ep',
            'InternVL2.5-2B (Feedback Vision LLM v1- 1ep)',
            'InternVL2_5-2B-Feedback-vision-llm-v1_1ep',
            '/home/ubuntu/workspace/elbiat/checkpoints/feedback_vision_llm_v1_1ep',
            true,
            'OpenGVLab/InternVL2_5-2B'
        )
        ON CONFLICT (name) DO UPDATE SET
            model_path = EXCLUDED.model_path,
            is_finetuned = EXCLUDED.is_finetuned,
            base_model = EXCLUDED.base_model;
    """))
    conn.commit()

    conn.execute(text("""
        INSERT INTO models (name, display_name, vlmeval_model, model_path, is_finetuned, base_model)
        VALUES (
            'internvl2_5_2b_feedback_vision_lora_10ep',
            'InternVL2.5-2B (Feedback Vision LoRa 10ep)',
            'InternVL2_5-2B-Feedback-vision-lora_10ep',
            '/home/ubuntu/workspace/elbiat/checkpoints/feedback_vision_lora_10ep',
            true,
            'OpenGVLab/InternVL2_5-2B'
        )
        ON CONFLICT (name) DO UPDATE SET
            model_path = EXCLUDED.model_path,
            is_finetuned = EXCLUDED.is_finetuned,
            base_model = EXCLUDED.base_model;
    """))
    conn.commit()

    conn.execute(text("""
        INSERT INTO models (name, display_name, vlmeval_model, model_path, is_finetuned, base_model)
        VALUES (
            'internvl2_5_2b_feedback_vision_lora_1ep',
            'InternVL2.5-2B (Feedback Vision LoRa 1ep)',
            'InternVL2_5-2B-Feedback-vision-lora_1ep',
            '/home/ubuntu/workspace/elbiat/checkpoints/feedback_vision_lora_1ep',
            true,
            'OpenGVLab/InternVL2_5-2B'
        )
        ON CONFLICT (name) DO UPDATE SET
            model_path = EXCLUDED.model_path,
            is_finetuned = EXCLUDED.is_finetuned,
            base_model = EXCLUDED.base_model;
    """))
    conn.commit()

    conn.execute(text("""
        INSERT INTO models (name, display_name, vlmeval_model, model_path, is_finetuned, base_model)
        VALUES (
            'internvl2_5_2b_feedback_vision_lora_100ep',
            'InternVL2.5-2B (Feedback Vision LoRa 100ep)',
            'InternVL2_5-2B-Feedback-vision-lora_100ep',
            '/home/ubuntu/workspace/elbiat/checkpoints/feedback_vision_lora_100ep',
            true,
            'OpenGVLab/InternVL2_5-2B'
        )
        ON CONFLICT (name) DO UPDATE SET
            model_path = EXCLUDED.model_path,
            is_finetuned = EXCLUDED.is_finetuned,
            base_model = EXCLUDED.base_model;
    """))
    conn.commit()

    conn.execute(text("""
        INSERT INTO models (name, display_name, vlmeval_model, model_path, is_finetuned, base_model)
        VALUES (
            'internvl2_5_2b_feedback_refined_v1_vision_lora_10ep',
            'InternVL2.5-2B (Feedback Refined v1 Vision LoRa 10ep)',
            'InternVL2_5-2B-Feedback-Refined-v1-vision-lora_10ep',
            '/home/ubuntu/workspace/elbiat/checkpoints/feedback_vision_lora_10ep',
            true,
            'OpenGVLab/InternVL2_5-2B'
        )
        ON CONFLICT (name) DO UPDATE SET
            model_path = EXCLUDED.model_path,
            is_finetuned = EXCLUDED.is_finetuned,
            base_model = EXCLUDED.base_model;
    """))
    conn.commit()

    conn.execute(text("""
        INSERT INTO models (name, display_name, vlmeval_model, model_path, is_finetuned, base_model)
        VALUES (
            'internvl2_5_2b_feedback_refined_v1_vision_lora_100ep',
            'InternVL2.5-2B (Feedback Refined v1 Vision LoRa 100ep)',
            'InternVL2_5-2B-Feedback-Refined-v1-vision-lora_100ep',
            '/home/ubuntu/workspace/elbiat/checkpoints/feedback_vision_lora_100ep',
            true,
            'OpenGVLab/InternVL2_5-2B'
        )
        ON CONFLICT (name) DO UPDATE SET
            model_path = EXCLUDED.model_path,
            is_finetuned = EXCLUDED.is_finetuned,
            base_model = EXCLUDED.base_model;
    """))
    conn.commit()

    conn.execute(text("""
        INSERT INTO models (name, display_name, vlmeval_model, model_path, is_finetuned, base_model)
        VALUES (
            'internvl2_5_8b_feedback_refined_v1_vision_lora_10ep',
            'InternVL2.5-8B (Feedback Refined v1 Vision LoRa 10ep)',
            'InternVL2_5-8B-Feedback-Refined-v1-vision-lora_10ep',
            '/home/ubuntu/workspace/elbiat/checkpoints/feedback_vision_lora_10ep_8b',
            true,
            'OpenGVLab/InternVL2_5-8B'
        )
        ON CONFLICT (name) DO UPDATE SET
            model_path = EXCLUDED.model_path,
            is_finetuned = EXCLUDED.is_finetuned,
            base_model = EXCLUDED.base_model;
    """))
    conn.commit()

    conn.execute(text("""
        INSERT INTO models (name, display_name, vlmeval_model, model_path, is_finetuned, base_model)
        VALUES (
            'internvl2_5_8b_feedback_refined_v1_vision_lora_1ep',
            'InternVL2.5-8B (Feedback Refined v1 Vision LoRa 1ep)',
            'InternVL2_5-8B-Feedback-Refined-v1-vision-lora_1ep',
            '/home/ubuntu/workspace/elbiat/checkpoints/feedback_vision_lora_1ep_8b',
            true,
            'OpenGVLab/InternVL2_5-8B'
        )
        ON CONFLICT (name) DO UPDATE SET
            model_path = EXCLUDED.model_path,
            is_finetuned = EXCLUDED.is_finetuned,
            base_model = EXCLUDED.base_model;
    """))
    conn.commit()

    print("Added finetuned model")
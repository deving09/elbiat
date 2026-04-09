"""
Wrapper for VLMEvalKit run.py that registers custom models first.
"""
import sys
from pathlib import Path

# Add paths
elbiat_root = Path(__file__).parent.parent
vlmeval_root = elbiat_root / "external" / "VLMEvalKit"

sys.path.insert(0, str(elbiat_root))
sys.path.insert(0, str(vlmeval_root))

# Register custom models BEFORE importing run
from vlm_custom_models import register
register()

# Register VPT models
from vpt.vpt_vlmeval import register_vpt_models
register_vpt_models()

# Register MOCHI dataset
from elbiat_datasets.mochi_vlmeval_dataset import register_mochi_datasets
register_mochi_datasets()


# Now run the main script
sys.argv[0] = str(vlmeval_root / "run.py")
exec(open(vlmeval_root / "run.py").read())
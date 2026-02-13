import sys
sys.path.append('/workspace/core-sum-project/src')
import numpy as np
from hierarchical_visual_pyramid import HierarchicalVisualPyramid
from cot_engine import CoTEngine

# Quick test
test_frames = [np.ones((224, 224, 3), dtype=np.uint8) * ((i//150) * 60) for i in range(600)]
hvp = HierarchicalVisualPyramid(device="cuda")
hierarchy = hvp.encode(test_frames)
cot_engine = CoTEngine()
result = cot_engine.summarize(hierarchy, target_frames=20)

print("\n" + "="*60)
print("COHERENCE CHECK DETAILS")
print("="*60)
print(f"Is coherent: {result.coherence_check['is_coherent']}")
print(f"\nIssues found:")
for issue in result.coherence_check.get('issues', []):
    print(f"  - {issue}")

print(f"\nSuggested additions: {len(result.coherence_check.get('suggested_additions', []))}")
for addition in result.coherence_check.get('suggested_additions', []):
    print(f"  Frame {addition['frame_index']}: {addition['reason']}")

print(f"\nFinal sequence length: {len(result.coherence_check['final_frame_sequence'])}")
print(f"Original + additions: {len(result.selected_frames)} + {len(result.coherence_check.get('suggested_additions', []))}")
print("="*60 + "\n")

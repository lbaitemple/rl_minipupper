"""Inspect the real shipped Mini Pupper policy input and output tensor shapes.

Run in the upstream mini_pupper_4_sim clone after CPU-mode installation:
    .venv/bin/python /path/to/inspect_mini_pupper_onnx.py policies/flat_omni
    .venv/bin/python /path/to/inspect_mini_pupper_onnx.py policies/stairs
"""
import argparse
from pathlib import Path
import json
import onnxruntime as ort


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('policy_dir', type=Path)
    args = ap.parse_args()
    folder = args.policy_dir
    model_file = folder / 'policy.onnx'
    if not model_file.is_file():
        ap.error(f'Missing {model_file}')
    sess = ort.InferenceSession(str(model_file), providers=['CPUExecutionProvider'])
    print('POLICY:', folder)
    for node in sess.get_inputs():
        print('input ', node.name, 'shape', node.shape, 'type', node.type)
    for node in sess.get_outputs():
        print('output', node.name, 'shape', node.shape, 'type', node.type)
    for file in ['config.json', 'manifest.json']:
        path = folder / file
        if path.exists():
            print(f'\n{file}:')
            print(json.dumps(json.loads(path.read_text()), indent=2)[:3500])
    # ONNX is an inference graph. The training-only critic may not be exported.


if __name__ == '__main__':
    main()

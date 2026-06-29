"""
Stage 2 — Disease Classifier HEF Compiler
Converts a YOLOv8n-cls ONNX export to a Hailo-8L HEF.
Usage
-----
    python compile/compile_stage2_percrop.py --crop eggplant --onnx <path/to/best.onnx>
Output
------
    compile/compiled_hef/stage2_<crop>_specialist.hef
Then copy/symlink into the models folder:
    cp compile/compiled_hef/stage2_eggplant_specialist.hef models/eggplant/classifier.hef
IMPORTANT — normalization
-------------------------
The /255 normalization is baked into the HEF via a .alls normalization() line.
Calibration data is fed as uint8 [0-255] (same range the baked layer expects).
At runtime, hailo_runner.py uses baked_norm=True → sends raw uint8 [0-255].
The HEF divides by 255 internally before the first conv layer.

This is consistent: calib range == runtime range == uint8 [0-255].

The previous approach (BAKE_NORMALIZATION=False, calib uint8, runtime float32 [0-1])
caused a calibration/runtime mismatch: the DFC calibrated quantization params for
[0-255] but inference sent [0-1], collapsing all predictions to one class at 100%.
"""
import os
import tempfile
import cv2
import numpy as np
import argparse
from hailo_sdk_client import ClientRunner
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# =============================================================================
HW_ARCH            = "hailo8l"
OUTPUT_DIR         = os.path.join(os.path.dirname(__file__), "compiled_hef")
CALIB_BATCH_SIZE   = 8
CALIB_TARGET_COUNT = 220
MIN_CALIB_IMAGES   = 64
INPUT_SIZE         = (224, 224)
NET_INPUT_SHAPES   = {"images": [1, 3, 224, 224]}
# Bake /255 into the HEF (same as Stage-1 detector).
# Runtime sends raw uint8 [0-255]; HEF divides by 255 internally.
# Calibration data must also be uint8 [0-255] to match.
BAKE_NORMALIZATION = True
# =============================================================================
def augment_calibration_dataset(images, target_count=CALIB_TARGET_COUNT):
    if len(images) >= target_count:
        print(f"  using {len(images)} real calibration images (no augmentation).")
        return images
    print(f"  only {len(images)} real images (< {target_count}); augmenting...")
    augmented = list(images)
    rng = np.random.default_rng(42)
    while len(augmented) < target_count:
        img = images[rng.integers(len(images))].copy()
        if rng.random() > 0.5:
            img = img[:, ::-1, :].copy()
        delta = int(rng.integers(-30, 31))
        img = np.clip(img.astype(np.int16) + delta, 0, 255).astype(np.uint8)
        scale = rng.uniform(0.8, 1.0)
        h, w = img.shape[:2]
        ch, cw = max(1, int(h * scale)), max(1, int(w * scale))
        y0 = int(rng.integers(0, h - ch + 1))
        x0 = int(rng.integers(0, w - cw + 1))
        img = cv2.resize(img[y0:y0+ch, x0:x0+cw], (w, h), interpolation=cv2.INTER_LINEAR)
        augmented.append(img)
    return np.array(augmented[:target_count], dtype=np.uint8)
def load_calibration_dataset(data_dir):
    """
    Load calibration images from a classification dataset directory.
    Accepts either:
      - A flat directory of images
      - An Ultralytics-format dir (train/<class>/<img>) — walks all subdirs
    """
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    images_list = []
    data_path = os.path.abspath(data_dir)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Calibration directory not found: {data_path}")
    for root, _, files in os.walk(data_path):
        for fname in sorted(files):
            if not fname.lower().endswith(valid_exts):
                continue
            img = cv2.imread(os.path.join(root, fname))
            if img is None:
                continue
            img = cv2.resize(img, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images_list.append(img)
    if not images_list:
        raise ValueError(f"No valid images found under: {data_path}")
    if len(images_list) < MIN_CALIB_IMAGES:
        print(f"  WARN: only {len(images_list)} images (< {MIN_CALIB_IMAGES} recommended).")
    print(f"  loaded {len(images_list)} calibration images from {data_path}")
    return augment_calibration_dataset(np.array(images_list, dtype=np.uint8))
def compile_classifier(crop: str, onnx_path: str, calib_dir: str):
    model_name = f"stage2_{crop}_specialist"
    print("=" * 60)
    print(f"Compiling Stage 2 classifier: {crop}")
    print(f"  source:    {onnx_path}")
    print(f"  target:    {HW_ARCH}")
    print(f"  norm baked: {BAKE_NORMALIZATION}  (runtime sends uint8 [0-255])")
    print("=" * 60)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    hef_out = os.path.join(OUTPUT_DIR, f"{model_name}.hef")
    if os.path.exists(hef_out):
        print(f"{model_name}.hef already exists — delete it to force a rebuild.")
        return hef_out
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX not found: {onnx_path}")
    runner = ClientRunner(hw_arch=HW_ARCH)
    print("  translating ONNX …")
    runner.translate_onnx_model(
        onnx_path,
        model_name,
        net_input_shapes=NET_INPUT_SHAPES,
    )
    alls = f"model_optimization_config(calibration, batch_size={CALIB_BATCH_SIZE})\n"
    alls += "performance_param(compiler_optimization_level=2)\n"
    if BAKE_NORMALIZATION:
        alls += "normalization([0,0,0],[255,255,255])\n"
        print("  baked normalization: normalization([0,0,0],[255,255,255])")

    # load_model_script() requires a file path, not an inline string
    with tempfile.NamedTemporaryFile(mode='w', suffix='.alls', delete=False) as tf:
        tf.write(alls)
        alls_path = tf.name
    try:
        runner.load_model_script(alls_path)
    finally:
        os.unlink(alls_path)
    calib_data = load_calibration_dataset(calib_dir)
    print("  full-precision pass …")
    runner.optimize_full_precision(calib_data=calib_data)
    print("  INT8 quantization …")
    runner.optimize(calib_data)
    print("  compiling …")
    hef_bytes = runner.compile()
    with open(hef_out, "wb") as f:
        f.write(hef_bytes)
    size_mb = os.path.getsize(hef_out) / 1e6
    print(f"\nHEF saved: {hef_out}  ({size_mb:.2f} MB)")
    print(f"""
Next steps:
  cp {hef_out} models/{crop}/classifier.hef
  (or update the symlink if one exists)
""")
    return hef_out
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Compile a Stage-2 classifier ONNX to Hailo-8L HEF"
    )
    ap.add_argument("--crop",  required=True,
                    choices=["eggplant", "rice", "corn", "tomato", "okra"],
                    help="Crop name")
    ap.add_argument("--onnx",  required=True,
                    help="Path to best.onnx from train_stage2_percrop.py")
    ap.add_argument("--calib", required=True,
                    help="Path to classification dataset used for calibration "
                         "(train/ subfolder or flat dir of images)")
    args = ap.parse_args()
    print("\n" + "#" * 60)
    print(f"   Stage 2 classifier HEF — {args.crop}")
    print("#" * 60 + "\n")
    try:
        compile_classifier(args.crop, args.onnx, args.calib)
        print("Done.")
    except (FileNotFoundError, ValueError) as e:
        print(f"\nError: {e}")
    except Exception as e:
        print(f"\nCompilation failure: {e}")
        raise

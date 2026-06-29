"""
Stage 2 — Disease Classifier HEF Compiler
Converts a YOLOv8n-cls ONNX export to a Hailo-8L HEF.
Usage
-----
    python compile/compile_stage2_percrop.py --crop eggplant --onnx <path/to/best.onnx> --calib <data_dir>
Output
------
    compile/compiled_hef/stage2_<crop>_specialist.hef
Then copy/symlink into the models folder:
    cp compile/compiled_hef/stage2_eggplant_specialist.hef models/eggplant/classifier.hef

NORMALIZATION
-------------
No normalization is baked into the HEF (no .alls normalization() line).
Calibration data is fed as float32 [0-1] (divided by 255 after loading).
At runtime, hailo_runner.py uses baked_norm=False → divides by 255 and sends
float32 [0-1] to the chip.

This is consistent: calib range == runtime range == float32 [0-1].

The original bug was: calib=uint8 [0-255] but runtime=float32 [0-1] → mismatch →
all predictions collapsed to one class at 100%.
"""
import os
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
        # jitter in float space
        delta = rng.uniform(-0.12, 0.12)
        img = np.clip(img + delta, 0.0, 1.0)
        scale = rng.uniform(0.8, 1.0)
        h, w = img.shape[:2]
        ch, cw = max(1, int(h * scale)), max(1, int(w * scale))
        y0 = int(rng.integers(0, h - ch + 1))
        x0 = int(rng.integers(0, w - cw + 1))
        img = cv2.resize(img[y0:y0+ch, x0:x0+cw], (w, h), interpolation=cv2.INTER_LINEAR)
        augmented.append(img)
    return np.array(augmented[:target_count], dtype=np.float32)


def load_calibration_dataset(data_dir):
    """
    Load calibration images as float32 [0-1] to match the runtime input range.
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
            img = img.astype(np.float32) / 255.0   # float32 [0-1] — matches runtime
            images_list.append(img)
    if not images_list:
        raise ValueError(f"No valid images found under: {data_path}")
    if len(images_list) < MIN_CALIB_IMAGES:
        print(f"  WARN: only {len(images_list)} images (< {MIN_CALIB_IMAGES} recommended).")
    print(f"  loaded {len(images_list)} calibration images  dtype=float32  range=[0,1]")
    return augment_calibration_dataset(np.array(images_list, dtype=np.float32))


def compile_classifier(crop: str, onnx_path: str, calib_dir: str):
    model_name = f"stage2_{crop}_specialist"
    print("=" * 60)
    print(f"Compiling Stage 2 classifier: {crop}")
    print(f"  source:    {onnx_path}")
    print(f"  target:    {HW_ARCH}")
    print(f"  norm:      none in HEF — calib & runtime both float32 [0-1]")
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
    runner.load_model_script(alls)
    calib_data = load_calibration_dataset(calib_dir)
    print(f"  calibration array: shape={calib_data.shape}  dtype={calib_data.dtype}")
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

Runtime: pipeline.py must use  baked_norm=False  for the Stage-2 engine.
         hailo_runner.py will divide by 255 and send float32 [0-1].
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

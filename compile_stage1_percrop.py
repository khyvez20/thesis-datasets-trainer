import os
import cv2
import numpy as np
import argparse
from hailo_sdk_client import ClientRunner

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# =============================================================================
# Single-crop Stage 1 HEF compiler (1 class per model).
# Run once per crop:
#   python compile_stage1_percrop.py --crop eggplant --onnx ./eggplant_best.onnx
#   python compile_stage1_percrop.py --crop okra     --onnx ./okra_best.onnx
# Produces: compiled_hef/stage1_<crop>.hef
#
# KEY DIFFERENCE vs the 4-class version: each detector is SINGLE-CLASS, so the
# cv3 (class) heads output 1 channel, not 4. On the Pi, the inference engine for
# this model must use NUM_CLASSES = 1 and reshape (-1, 1).
# =============================================================================

HW_ARCH            = "hailo8l"
CALIB_DIR          = "calibration_images"   # per-crop calib images (get_calib_images.py)
OUTPUT_DIR         = "compiled_hef"
CALIB_BATCH_SIZE   = 8
CALIB_TARGET_COUNT = 220
MIN_CALIB_IMAGES   = 64
NET_INPUT_SHAPES   = {"images": [1, 3, 640, 640]}
BAKE_NORMALIZATION = True   # bake /255 -> feed RAW uint8 on the Pi (no /255 in runtime!)

# Single-class YOLOv8 end-nodes. Node NAMES are the same as any YOLOv8; only the
# cv3 class-channel width changes (1 for a single-crop detector).
DETECTOR_END_NODES = [
    "/model.22/cv2.0/cv2.0.2/Conv",   # P3 box   (80x80x64)
    "/model.22/cv3.0/cv3.0.2/Conv",   # P3 class (80x80x1)
    "/model.22/cv2.1/cv2.1.2/Conv",   # P4 box   (40x40x64)
    "/model.22/cv3.1/cv3.1.2/Conv",   # P4 class (40x40x1)
    "/model.22/cv2.2/cv2.2.2/Conv",   # P5 box   (20x20x64)
    "/model.22/cv3.2/cv3.2.2/Conv",   # P5 class (20x20x1)
]


def verify_end_nodes(onnx_path, end_nodes):
    try:
        import onnx
    except ImportError:
        print("  (pip install onnx to enable end-node verification)")
        return
    model = onnx.load(onnx_path)
    names = {n.name for n in model.graph.node}
    missing = [e for e in end_nodes if e not in names]
    if missing:
        print("\nEND-NODE MISMATCH — not found in ONNX:")
        for m in missing:
            print("   ", m)
        head = [n.name for n in model.graph.node
                if n.op_type == "Conv" and "model.22" in n.name]
        print("\nAvailable head Conv nodes:")
        for h in head:
            print("   ", h)
        raise ValueError("End-node names do not match the ONNX graph.")
    print(f"  end-nodes verified ({len(end_nodes)} found).")


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
        delta = int(rng.integers(-40, 41))
        img = np.clip(img.astype(np.int16) + delta, 0, 255).astype(np.uint8)
        scale = rng.uniform(0.75, 1.0)
        h, w = img.shape[:2]
        ch, cw = max(1, int(h * scale)), max(1, int(w * scale))
        y0 = int(rng.integers(0, h - ch + 1))
        x0 = int(rng.integers(0, w - cw + 1))
        img = cv2.resize(img[y0:y0+ch, x0:x0+cw], (w, h), interpolation=cv2.INTER_LINEAR)
        augmented.append(img)
    return np.array(augmented[:target_count], dtype=np.uint8)


def load_calibration_dataset(directory_path, target_size=(640, 640)):
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Calibration directory not found: {directory_path}")
    images_list, skipped = [], 0
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    for filename in sorted(os.listdir(directory_path)):
        if not filename.lower().endswith(valid_exts):
            continue
        img = cv2.imread(os.path.join(directory_path, filename))
        if img is None:
            skipped += 1
            continue
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images_list.append(img)
    if skipped:
        print(f"  skipped {skipped} unreadable image(s).")
    if not images_list:
        raise ValueError(f"No valid calibration images in: {directory_path}")
    if len(images_list) < MIN_CALIB_IMAGES:
        print(f"  WARN: only {len(images_list)} raw images (< {MIN_CALIB_IMAGES}).")
    print(f"  loaded {len(images_list)} raw calibration images.")
    return augment_calibration_dataset(np.array(images_list, dtype=np.uint8))


def convert_detector(crop, onnx_path, calib_dir):
    model_name = f"stage1_{crop}"
    print("=" * 60)
    print(f"Compiling single-crop detector: {crop}")
    print(f"  source: {onnx_path}   target: {HW_ARCH}")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    hef_path = os.path.join(OUTPUT_DIR, f"{model_name}.hef")
    if os.path.exists(hef_path):
        print(f"{model_name}.hef already exists — delete to force rebuild.")
        return
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX not found: {onnx_path}")

    verify_end_nodes(onnx_path, DETECTOR_END_NODES)
    runner = ClientRunner(hw_arch=HW_ARCH)
    print(f"  translating with {len(DETECTOR_END_NODES)} end-nodes...")
    runner.translate_onnx_model(
        onnx_path, model_name,
        start_node_names=["images"],
        end_node_names=DETECTOR_END_NODES,
        net_input_shapes=NET_INPUT_SHAPES,
    )

    alls = ""
    if BAKE_NORMALIZATION:
        alls += "normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])\n"
    alls += f"model_optimization_config(calibration, batch_size={CALIB_BATCH_SIZE})\n"
    alls += "performance_param(compiler_optimization_level=2)\n"
    runner.load_model_script(alls)
    print("  model script loaded" + (" (/255 baked)." if BAKE_NORMALIZATION else "."))

    calib_data = load_calibration_dataset(calib_dir)
    print("  full-precision pass...")
    runner.optimize_full_precision(calib_data=calib_data)
    print("  INT8 quantization...")
    runner.optimize(calib_data)
    print("  compiling...")
    hef_data = runner.compile()
    with open(hef_path, "wb") as f:
        f.write(hef_data)
    print(f"HEF saved: {hef_path}  ({os.path.getsize(hef_path)/1e6:.2f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", required=True,
                    choices=["eggplant", "tomato", "rice", "corn", "okra"])
    ap.add_argument("--onnx", required=True, help="path to that crop's best.onnx")
    ap.add_argument("--calib", default=CALIB_DIR, help="calibration images dir for this crop")
    args = ap.parse_args()
    print("\n" + "#" * 60)
    print(f"   Stage 1 single-crop HEF — {args.crop}  (1 class)")
    print("#" * 60 + "\n")
    try:
        convert_detector(args.crop, args.onnx, args.calib)
        print("\nDone.")
    except (FileNotFoundError, ValueError) as e:
        print(f"\nError: {e}")
    except Exception as e:
        print(f"\nCompilation failure: {e}")
        raise
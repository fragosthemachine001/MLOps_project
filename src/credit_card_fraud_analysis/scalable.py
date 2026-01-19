import torch
import torch.nn.utils.prune as prune
from onnxruntime.quantization import quantize_dynamic, QuantType
from ptflops import get_model_complexity_info
from credit_card_fraud_analysis.lightning_module import LitAutoEncoder
from pathlib import Path

# Setup paths
MODELS_DIR = Path("models")
ONNX_PATH = MODELS_DIR / "model_fp32.onnx"
QUANT_ONNX_PATH = MODELS_DIR / "optimized_model.onnx"


def check_complexity(model):
    macs, params = get_model_complexity_info(
        model, (28,),
        as_strings=True,
        print_per_layer_stat=False,
        verbose=False
    )
    print(f"--- Architecture Complexity Report ---")
    print(f"Computational complexity (MACs): {macs}")
    print(f"Number of parameters: {params}")
    print(f"--------------------------------------")


def optimize():
    # 1. Load the most recent checkpoint
    ckpt_files = list(MODELS_DIR.glob("*.ckpt"))
    if not ckpt_files:
        raise FileNotFoundError(f"No .ckpt files found in {MODELS_DIR}")
    CKPT_PATH = max(ckpt_files, key=lambda p: p.stat().st_mtime)
    print(f"Using checkpoint: {CKPT_PATH}")

    model = LitAutoEncoder.load_from_checkpoint(CKPT_PATH)
    model.eval()
    check_complexity(model)

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=0.2)
            prune.remove(module, 'weight')
    print("Pruning complete: 20% of weights zeroed.")

    dummy_input = torch.randn(1, 28)
    torch.onnx.export(
        model, dummy_input, ONNX_PATH,
        export_params=True, opset_version=11,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Model exported to {ONNX_PATH}")

    quantize_dynamic(
        model_input=ONNX_PATH,
        model_output=QUANT_ONNX_PATH,
        weight_type=QuantType.QUInt8
    )
    print(f"Quantization complete! Final model: {QUANT_ONNX_PATH}")


if __name__ == "__main__":
    optimize()
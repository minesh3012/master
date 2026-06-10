import onnx
from onnx import shape_inference

model_path = r"C:\Trading\Projects\ES_AI_Project\models\catboost\catboost_LongSuccess.onnx"

print("Loading ONNX model...")
model = onnx.load(model_path)

print("Model IR version:", model.ir_version)
print("Producer:", model.producer_name)
print("Opset:", model.opset_import)

print("\n=== OUTPUTS ===")
for out in model.graph.output:
    print("Name:", out.name)
    print("Type:", out.type)
    print()

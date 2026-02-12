from ai_edge_litert.interpreter import Interpreter
path = "./aliked-n16_fp16.tflite"

interpreter = Interpreter(model_path=path)
interpreter.allocate_tensors()

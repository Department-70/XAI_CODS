import onnx
from onnx_tf.backend import prepare
onnx_model = onnx.load("./Onnx/Model_50_gen.onnx")
tf_rep = prepare(onnx_model)
tf_rep.export_graph("./results/Onnx.pb")
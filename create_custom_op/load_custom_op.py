## let's try to load the created custom op

import onnxruntime as ort
import numpy as np

def create_model(path):
    # create a model containing the custom op from the library
    # CustomOpOne
    #   inputs: two float arrays x and y
    #   output: out[i] = x[i] + y[i]
    from onnx import helper, save
    from onnx import TensorProto

    # inputs
    x_input = helper.make_tensor_value_info('x', TensorProto.FLOAT, [5,])
    y_input = helper.make_tensor_value_info('y', TensorProto.FLOAT, [5,])

    # output
    ## TODO: do I need to explicitly init the output tensor too?
    output = helper.make_tensor_value_info('out', TensorProto.FLOAT, [5, ])
    node = helper.make_node(
        "CustomOpOne",
        ['x', 'y'],
        ['out'],
        domain="test.customop"
    )

    graph = helper.make_graph(
        [node, ],
        "test-model",
        [x_input, y_input],
        [output, ]
    )

    model = helper.make_model(graph, producer_name="customop-example")
    save(model, path)
    # return model

if __name__ == "__main__":
    # shared_library = "./libcustomop.so"
    shared_library = "./build/libcustomop.so"

    session_options = ort.SessionOptions()
    session_options.register_custom_ops_library(shared_library)

    onnx_model = "model.onnx"
    create_model(onnx_model)

    session = ort.InferenceSession(onnx_model, sess_options=session_options)
    x_data = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    y_data = np.array([1, 2, 3, 4, 5], dtype=np.float32)

    input_x_name = session.get_inputs()[0].name
    input_y_name = session.get_inputs()[1].name
    output_name = session.get_outputs()[0].name
    res = session.run([output_name], {input_x_name: x_data, input_y_name: y_data})
    print(res[0])
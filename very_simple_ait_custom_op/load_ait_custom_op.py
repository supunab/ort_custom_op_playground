from statistics import mode
from tabnanny import check
import onnxruntime as ort
import numpy as np

"""
The model is a single linear
input: 4 x 10
linear weights: 10 x 10
output: 4 x 10
"""
batch_size = 4
input_size = 10
hidden_size = 10

def create_model(path):
    from onnx import helper, save
    from onnx import TensorProto
    from onnx import checker

    # inputs
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT16, shape=(batch_size, input_size))
    weight = helper.make_tensor_value_info('weight', TensorProto.FLOAT16, shape=(input_size, hidden_size))

    # output
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT16, shape=(batch_size, hidden_size))
    node = helper.make_node(
        "AITModelOp",
        ['input', 'weight'],
        ['output'],
        name="linear",
        domain="ait.customop"
    )

    graph = helper.make_graph(
        [node, ],
        "test-model",
        [input, weight],
        [output, ]
    )

    model = helper.make_model(graph, producer_name="ait-customop-example")
    save(model, path)


def torch_output(input_data, weight_data):
    import torch
    import torch.nn as nn
    
    class VerySimpleModel(nn.Module):
        def __init__(self, hidden_size) -> None:
            super().__init__()
            self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
        
        def forward(self, input):
            hidden_states = self.linear(input)
            return hidden_states

    model = VerySimpleModel(hidden_size).half().cuda().eval()
    input_torch = torch.from_numpy(input_data).cuda()
    assert(input_torch.dtype == torch.float16) # make sure data type is preserved
    weight_torch = torch.from_numpy(weight_data).cuda()
    assert(weight_torch.dtype == torch.float16)

    model.linear.weight.data = weight_torch
    output = model(input_torch)
    return output


if __name__ == "__main__":
    shared_library = "./test.so" # was compiled from the docker container. Should be fine since it's the same OS?

    session_options = ort.SessionOptions()
    session_options.register_custom_ops_library(shared_library)
    
    onnx_model = "model.onnx"
    create_model(onnx_model)

    session = ort.InferenceSession(onnx_model, sess_options=session_options, providers=["CUDAExecutionProvider"])

    input_data = np.random.rand(batch_size, input_size).astype(np.float16)
    weight_data = np.random.rand(input_size, hidden_size).astype(np.float16)
    input_ortvalue = ort.OrtValue.ortvalue_from_numpy(input_data, "cuda", 0)
    weight_ortvalue = ort.OrtValue.ortvalue_from_numpy(weight_data, "cuda", 0)

    output_ortvalue = ort.OrtValue.ortvalue_from_shape_and_type((batch_size, hidden_size), np.float16, "cuda", 0)

    io_binding = session.io_binding()

    input_name = session.get_inputs()[0].name
    io_binding.bind_input(name=input_name, device_type=input_ortvalue.device_name(), device_id=0, 
                        element_type=np.float16, shape=input_ortvalue.shape(), buffer_ptr=input_ortvalue.data_ptr())

    weight_name = session.get_inputs()[1].name
    io_binding.bind_input(name=weight_name, device_type=weight_ortvalue.device_name(), device_id=0, 
                        element_type=np.float16, shape=weight_ortvalue.shape(), buffer_ptr=weight_ortvalue.data_ptr())

    output_name = session.get_outputs()[0].name
    io_binding.bind_output(name=output_name, device_type=output_ortvalue.device_name(), device_id=0, 
                        element_type=np.float16, shape=output_ortvalue.shape(), buffer_ptr=output_ortvalue.data_ptr())

    session.run_with_iobinding(io_binding)
    ait_output = output_ortvalue.numpy()
    torch_output = torch_output(input_data, weight_data).cpu().detach().numpy()

    equal = np.allclose(ait_output, torch_output, atol=1e-1)

    # print(ait_output)
    # print(torch_output)

    if equal:
        print("Outputs matched! (to a 0.1 tolerenace")
    else:
        print("ERROR! outputs are different!")

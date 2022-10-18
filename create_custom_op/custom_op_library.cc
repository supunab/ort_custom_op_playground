#include "custom_op_library.h"

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include <vector>
#include <cmath>
#include <mutex>
#include <unistd.h>

static const char *c_OpDomain = "test.customop";

// TODO: why this is needed? what needs to be done?
struct OrtCustomOpDomainDeleter {
  explicit OrtCustomOpDomainDeleter(const OrtApi *ort_api) {
    ort_api_ = ort_api;
  }
  void operator()(OrtCustomOpDomain *domain) const {
    ort_api_->ReleaseCustomOpDomain(domain);
  }

  const OrtApi *ort_api_;
};

using OrtCustomOpDomainUniquePtr = std::unique_ptr<OrtCustomOpDomain, OrtCustomOpDomainDeleter>;
static std::vector<OrtCustomOpDomainUniquePtr> ort_custom_op_domain_container;
static std::mutex ort_custom_op_domain_mutex;

// TODO: what is this? why this is needed?
static void AddOrtCustomOpDomainToContainer(OrtCustomOpDomain *domain, const OrtApi *ort_api)
{
  std::lock_guard<std::mutex> lock(ort_custom_op_domain_mutex);
  auto ptr = std::unique_ptr<OrtCustomOpDomain, OrtCustomOpDomainDeleter>(domain, OrtCustomOpDomainDeleter(ort_api));
  ort_custom_op_domain_container.push_back(std::move(ptr));
}

// this basically wraps the shape of the tensor
// gets shape from the info and store it here and releases the info object
//          essentially makes a copy -- maybe this is needed due to memory deallocation 
struct OrtTensorDimensions : std::vector<int64_t> {
  OrtTensorDimensions(Ort::CustomOpApi ort, const OrtValue* value) {
    OrtTensorTypeAndShapeInfo* info = ort.GetTensorTypeAndShape(value);
    std::vector<int64_t>::operator=(ort.GetTensorShape(info));
    ort.ReleaseTensorTypeAndShapeInfo(info);
  }
};

// create the kernel for custom op
struct KernelOne {
  KernelOne(const OrtApi& api)
      : ort_(api) {
  }

  void Compute(OrtKernelContext* context) {
    // Setup inputs
    const OrtValue* input_X = ort_.KernelContext_GetInput(context, 0);
    const OrtValue* input_Y = ort_.KernelContext_GetInput(context, 1);
    const float* X = ort_.GetTensorData<float>(input_X);
    const float* Y = ort_.GetTensorData<float>(input_Y);

    // Setup output
    OrtTensorDimensions dimensions(ort_, input_X);

    OrtValue* output = ort_.KernelContext_GetOutput(context, 0, dimensions.data(), dimensions.size());
    float* out = ort_.GetTensorMutableData<float>(output);

    OrtTensorTypeAndShapeInfo* output_info = ort_.GetTensorTypeAndShape(output);
    int64_t size = ort_.GetTensorShapeElementCount(output_info);
    ort_.ReleaseTensorTypeAndShapeInfo(output_info);

    // Do computation
#ifdef USE_CUDA
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ort_.KernelContext_GetGPUComputeStream(context));
    cuda_add(size, out, X, Y, stream);
#else
    for (int64_t i = 0; i < size; i++) {
      out[i] = X[i] + Y[i];
    }
#endif
  }

 private:
  Ort::CustomOpApi ort_;
};

// define the CustomOp
struct CustomOpOne : Ort::CustomOpBase<CustomOpOne, KernelOne> {
  void* CreateKernel(const OrtApi& api, const OrtKernelInfo* /* info */) const {
    return new KernelOne(api);
  };

  const char* GetName() const { return "CustomOpOne"; };

#ifdef USE_CUDA
  const char* GetExecutionProviderType() const { return "CUDAExecutionProvider"; };
#endif

  size_t GetInputTypeCount() const { return 2; };
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };

  size_t GetOutputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };

} c_CustomOpOne;

// this is the entry point that gets called when this is loaded as a shared library
OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api) {
  OrtCustomOpDomain* domain = nullptr;
  const OrtApi* ortApi = api->GetApi(ORT_API_VERSION);

  if (auto status = ortApi->CreateCustomOpDomain(c_OpDomain, &domain)) {
    return status;
  }

  AddOrtCustomOpDomainToContainer(domain, ortApi);

  if (auto status = ortApi->CustomOpDomain_Add(domain, &c_CustomOpOne)) {
    return status;
  }

  return ortApi->AddCustomOpDomain(options, domain);
}

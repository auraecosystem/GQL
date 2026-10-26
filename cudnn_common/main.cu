#include "cuda_common.hpp" // CHECK_CUDA
#include "cudnn_common.hpp" // CHECK_CUDNN

#include <cuda_runtime.h>
#include <cudnn.h>

#include <iostream>
#include <vector>
#include <numeric>
#include <array>
#include <cassert>

constexpr int N = 1, C = 32, H = 416, W = 416;
constexpr int K = 32, M = 1, D = 1, S = 1;
constexpr int G = 1;

constexpr int P = M / 2;

const auto MAP_H = (H + 2 * P - ((M - 1) * D + 1)) / S + 1;
const auto MAP_W = (W + 2 * P - ((M - 1) * D + 1)) / S + 1;

cudnnBackendDescriptor_t createTensorDescriptor(std::vector<int64_t> dims, cudnnDataType_t dataType, bool isVirtual, int64_t uid)
{
    int64_t alignment = 256; // TODO
    int64_t rank = dims.size();

    std::vector<int64_t> strides(rank);
    strides.back() = 1;
    /* WHAT WE HAVE NOW:
     * strides[-1] = 1
     * strides[-2] = garbage
     * strides[-3] = garbage
     * strides[-4] = garbage
     * ...
     */

    std::copy(dims.begin() + 1, dims.end(), strides.begin());
    /* WHAT WE HAVE NOW:
     * strides[-1] = 1
     * strides[-2] = dim[-1]
     * strides[-3] = dim[-2]
     * strides[-4] = dim[-3]
     * ...
     */

    std::partial_sum(strides.rbegin(), strides.rend(), strides.rbegin(), std::multiplies<int>());
    /* WHAT WE HAVE NOW:
     * strides[-1] = 1
     * strides[-2] = strides[-1] * dim[-1]
     * strides[-3] = strides[-2] * dim[-2]
     * strides[-4] = strides[-3] * dim[-3]
     * ...
     */

    cudnnBackendDescriptor_t tensorDesc;
    CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &tensorDesc));
    CHECK_CUDNN(cudnnBackendSetAttribute(tensorDesc, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &dataType));
    CHECK_CUDNN(cudnnBackendSetAttribute(tensorDesc, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &alignment));
    CHECK_CUDNN(cudnnBackendSetAttribute(tensorDesc, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, rank, dims.data()));
    CHECK_CUDNN(cudnnBackendSetAttribute(tensorDesc, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, rank, strides.data()));
    CHECK_CUDNN(cudnnBackendSetAttribute(tensorDesc, CUDNN_ATTR_TENSOR_IS_VIRTUAL, CUDNN_TYPE_BOOLEAN, 1, &isVirtual));
    CHECK_CUDNN(cudnnBackendSetAttribute(tensorDesc, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &uid));
    CHECK_CUDNN(cudnnBackendFinalize(tensorDesc));
    return tensorDesc;
}

cudnnBackendDescriptor_t createConvolutionDescriptor(
    cudnnDataType_t computeType,
    std::vector<int64_t> dilations, std::vector<int64_t> strides,
    std::vector<int64_t> padding_left, std::vector<int64_t> padding_right)
{
    cudnnConvolutionMode_t convMode = CUDNN_CROSS_CORRELATION;

    const int64_t order = dilations.size();
    assert(strides.size() == order);
    assert(padding_left.size() == order);
    assert(padding_right.size() == order);

    cudnnBackendDescriptor_t convDesc;
    CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR, &convDesc));
    CHECK_CUDNN(cudnnBackendSetAttribute(convDesc, CUDNN_ATTR_CONVOLUTION_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &computeType));
    CHECK_CUDNN(cudnnBackendSetAttribute(convDesc, CUDNN_ATTR_CONVOLUTION_CONV_MODE, CUDNN_TYPE_CONVOLUTION_MODE, 1, &convMode));
    CHECK_CUDNN(cudnnBackendSetAttribute(convDesc, CUDNN_ATTR_CONVOLUTION_DILATIONS, CUDNN_TYPE_INT64, dilations.size(), dilations.data()));
    CHECK_CUDNN(cudnnBackendSetAttribute(convDesc, CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES, CUDNN_TYPE_INT64, strides.size(), strides.data()));
    CHECK_CUDNN(cudnnBackendSetAttribute(convDesc, CUDNN_ATTR_CONVOLUTION_POST_PADDINGS, CUDNN_TYPE_INT64, padding_left.size(), padding_left.data()));
    CHECK_CUDNN(cudnnBackendSetAttribute(convDesc, CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS, CUDNN_TYPE_INT64, padding_right.size(), padding_right.data()));
    CHECK_CUDNN(cudnnBackendSetAttribute(convDesc, CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS, CUDNN_TYPE_INT64, 1, &order));
    CHECK_CUDNN(cudnnBackendFinalize(convDesc));
    return convDesc;
}

cudnnBackendDescriptor_t createConvolutionOperationDescriptor(
    cudnnBackendDescriptor_t convDesc,
    cudnnBackendDescriptor_t inputDesc, cudnnBackendDescriptor_t filterDesc, cudnnBackendDescriptor_t outputDesc,
    double alpha = 1.0, double beta = 0.0)
{
    cudnnBackendDescriptor_t opDesc;
    CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR, &opDesc));
    CHECK_CUDNN(cudnnBackendSetAttribute(opDesc, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, convDesc));
    CHECK_CUDNN(cudnnBackendSetAttribute(opDesc, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, inputDesc));
    CHECK_CUDNN(cudnnBackendSetAttribute(opDesc, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, filterDesc));
    CHECK_CUDNN(cudnnBackendSetAttribute(opDesc, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, outputDesc));
    CHECK_CUDNN(cudnnBackendSetAttribute(opDesc, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA, CUDNN_TYPE_DOUBLE, 1, &alpha));
    CHECK_CUDNN(cudnnBackendSetAttribute(opDesc, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA, CUDNN_TYPE_DOUBLE, 1, &beta));
    CHECK_CUDNN(cudnnBackendFinalize(opDesc));
    return opDesc;
}

template <typename... Args>
cudnnBackendDescriptor_t createOpSet(cudnnHandle_t handle, Args... ops)
{
    constexpr int64_t numOps = sizeof...(Args);
    std::array<cudnnBackendDescriptor_t, numOps> opsArr = {ops...};

    cudnnBackendDescriptor_t opSetDesc;
    CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &opSetDesc));
    CHECK_CUDNN(cudnnBackendSetAttribute(opSetDesc, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, 1, &handle));
    CHECK_CUDNN(cudnnBackendSetAttribute(opSetDesc, CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, numOps, opsArr.data()));
    CHECK_CUDNN(cudnnBackendFinalize(opSetDesc));
    return opSetDesc;
}

std::vector<cudnnBackendNumericalNote_t> getNumericalNotes(cudnnBackendDescriptor_t engine)
{
    int64_t returnedNoteCount = 0;
    std::vector<cudnnBackendNumericalNote_t> notes(10);
    CHECK_CUDNN(cudnnBackendGetAttribute(engine, CUDNN_ATTR_ENGINE_NUMERICAL_NOTE, CUDNN_TYPE_NUMERICAL_NOTE, notes.size(), &returnedNoteCount, notes.data()));
    notes.resize(returnedNoteCount);
    return notes; 
}

cudnnBackendDescriptor_t createEngineHeuristicsDescriptor(cudnnBackendDescriptor_t opSet, cudnnBackendHeurMode_t heurMode)
{
    cudnnBackendDescriptor_t engHeur;
    CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, &engHeur));
    CHECK_CUDNN(cudnnBackendSetAttribute(engHeur, CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, opSet));
    CHECK_CUDNN(cudnnBackendSetAttribute(engHeur, CUDNN_ATTR_ENGINEHEUR_MODE, CUDNN_TYPE_HEUR_MODE, 1, &heurMode));
    CHECK_CUDNN(cudnnBackendFinalize(engHeur));
    return engHeur;
}

std::vector<cudnnBackendDescriptor_t> getEngineConfigs(cudnnBackendDescriptor_t engHeur)
{
    constexpr int LIMIT = 10;
    int64_t returnedConfigsCount = 0;
    std::vector<cudnnBackendDescriptor_t> engConfigs(10);
    for (int i = 0; i < LIMIT; i++)
        CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR, &engConfigs[i]));
    CHECK_CUDNN(cudnnBackendGetAttribute(engHeur, CUDNN_ATTR_ENGINEHEUR_RESULTS, CUDNN_TYPE_BACKEND_DESCRIPTOR, engConfigs.size(), &returnedConfigsCount, engConfigs.data()));
    for (int i = returnedConfigsCount; i < LIMIT; i++)
        CHECK_CUDNN(cudnnBackendDestroyDescriptor(engConfigs[i]));
    engConfigs.resize(returnedConfigsCount);
    return engConfigs;
}

cudnnBackendDescriptor_t createExecutionPlan(cudnnHandle_t handle, cudnnBackendDescriptor_t engConfig)
{
    cudnnBackendDescriptor_t execPlan;
    CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &execPlan));
    CHECK_CUDNN(cudnnBackendSetAttribute(execPlan, CUDNN_ATTR_EXECUTION_PLAN_HANDLE, CUDNN_TYPE_HANDLE, 1, &handle));
    CHECK_CUDNN(cudnnBackendSetAttribute(execPlan, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, engConfig));
    CHECK_CUDNN(cudnnBackendFinalize(execPlan));
    return execPlan;
}

cudnnBackendDescriptor_t createVariantPack(std::vector<int64_t> uids, std::vector<void*> devPtrs, void* workspace)
{
    cudnnBackendDescriptor_t varPack;
    CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &varPack));    
    CHECK_CUDNN(cudnnBackendSetAttribute(varPack, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, uids.size(), uids.data()));
    CHECK_CUDNN(cudnnBackendSetAttribute(varPack, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, devPtrs.size(), devPtrs.data()));
    CHECK_CUDNN(cudnnBackendSetAttribute(varPack, CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, 1, workspace));
    CHECK_CUDNN(cudnnBackendFinalize(varPack));
    return varPack;
}

int main ()
{
    float *input_d = nullptr;
    {
        CHECK_CUDA(cudaMalloc(&input_d, N * C * H * W * sizeof(float)));

        float *input_h = new float[N * C * H * W];
        for (int i = 0; i < N * C * H * W; i++)
            input_h[i] = (i % 1024) / 1024.0;
        CHECK_CUDA(cudaMemcpy(input_d, input_h, N * C * H * W * sizeof(float), cudaMemcpyHostToDevice));
    }

    float *filters_d = nullptr;
    {
        CHECK_CUDA(cudaMalloc(&filters_d, K * C * M * M * sizeof(float)));

        float *filters_h = new float[K * C * M * M];
        for (int i = 0; i < K * C * M * M; i++)
        filters_h[i] = (i % 128) / 128.0;
        CHECK_CUDA(cudaMemcpy(filters_d, filters_h, K * C * M * M * sizeof(float), cudaMemcpyHostToDevice));
    }

    constexpr int output_size = N * K * MAP_H * MAP_W;

    float *output_d = nullptr;
    CHECK_CUDA(cudaMalloc(&output_d, output_size * sizeof(float)));

    cudnnHandle_t handle;
    CHECK_CUDNN(cudnnCreate(&handle));
    
    auto inputTensorDesc = createTensorDescriptor({N, C, H, W}, CUDNN_DATA_FLOAT, false, 'I');
    auto filtersTensorDesc = createTensorDescriptor({K, C, M, M}, CUDNN_DATA_FLOAT, false, 'W');
    auto outputTensorDesc = createTensorDescriptor({N, K, MAP_H, MAP_W}, CUDNN_DATA_FLOAT, false, 'O');

    auto convDesc = createConvolutionDescriptor(CUDNN_DATA_FLOAT, {D, D}, {S, S}, {P, P}, {P, P});
    auto opDesc = createConvolutionOperationDescriptor(convDesc, inputTensorDesc, filtersTensorDesc, outputTensorDesc);
    
    auto opSetDesc = createOpSet(handle, opDesc);
    auto engHeur = createEngineHeuristicsDescriptor(opSetDesc, CUDNN_HEUR_MODE_INSTANT);
    auto engConfigs = getEngineConfigs(engHeur);

    for (auto config : engConfigs)
    {
        for(auto note : getNumericalNotes(config))
        {
            std::cout << "Engine Configuration Entry: ";
            switch(note)
            {
                case CUDNN_NUMERICAL_NOTE_TENSOR_CORE: std::cout << "\tTensor Cores\n"; break;
                case CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS: std::cout << "\tDown Convert Inputs\n"; break;
                case CUDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION: std::cout <<"\tReducedPrecisionReduction\n"; break;
                case CUDNN_NUMERICAL_NOTE_FFT: std::cout <<"\tFFT\n"; break;
                case CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC: std::cout <<"\tNon Deterministic\n"; break;
                case CUDNN_NUMERICAL_NOTE_WINOGRAD: std::cout <<"\tWinograd\n"; break;
                case CUDNN_NUMERICAL_NOTE_TYPE_COUNT: std::cout <<"\tType Count\n"; break;
                default:
                    std::cout <<"\tUnknown Note\n";
                    break;
            }
            std::cout << std::endl;
        }
    }

    assert(engConfigs.size() >= 1);
    auto execPlan = createExecutionPlan(handle, engConfigs[0]);    

    void* workspace_d = nullptr; // TODO

    auto varPack = createVariantPack({'I', 'W', 'O'}, {input_d, filters_d, output_d}, workspace_d);
    CHECK_CUDNN(cudnnBackendExecute(handle, execPlan, varPack));
    return 0;
}

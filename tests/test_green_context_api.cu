/**
 * 测试 Green Context API 是否可用
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK_CU(call) do { \
    CUresult err = call; \
    if (err != CUDA_SUCCESS) { \
        const char* errStr; \
        cuGetErrorString(err, &errStr); \
        printf("CUDA Driver API error at %s:%d: %s\n", __FILE__, __LINE__, errStr); \
        return 1; \
    } \
} while(0)

int main() {
    printf("Testing Green Context API...\n");

    // Step 1: 初始化 CUDA Driver API
    CHECK_CU(cuInit(0));
    printf("✓ cuInit succeeded\n");

    CUdevice device;
    CHECK_CU(cuDeviceGet(&device, 0));
    printf("✓ cuDeviceGet succeeded\n");

    // 获取设备名称和计算能力
    char name[256];
    cuDeviceGetName(name, sizeof(name), device);
    int major, minor;
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
    printf("  Device: %s (sm_%d%d)\n", name, major, minor);

    // Step 2: 获取 SM 资源
    CUdevResource full_sm_resource;
    CUresult result = cuDeviceGetDevResource(device, &full_sm_resource, CU_DEV_RESOURCE_TYPE_SM);
    if (result != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(result, &errStr);
        printf("✗ cuDeviceGetDevResource failed: %s\n", errStr);
        printf("  Green Context may not be supported on this GPU/driver\n");
        return 1;
    }
    printf("✓ cuDeviceGetDevResource succeeded\n");

    // Step 3: 分割 SM 资源
    unsigned int hp_sms = 64;
    unsigned int be_sms = 48;

    CUdevResource sm_groups[2];
    CUdevResource remaining;
    unsigned int hp_groups = 1;

    result = cuDevSmResourceSplitByCount(&sm_groups[0], &hp_groups, &full_sm_resource, &remaining, 0, hp_sms);
    if (result != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(result, &errStr);
        printf("✗ cuDevSmResourceSplitByCount (HP) failed: %s\n", errStr);
        return 1;
    }
    printf("✓ Split HP partition: %u SMs\n", hp_sms);

    unsigned int be_groups = 1;
    result = cuDevSmResourceSplitByCount(&sm_groups[1], &be_groups, &remaining, NULL, 0, be_sms);
    if (result != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(result, &errStr);
        printf("✗ cuDevSmResourceSplitByCount (BE) failed: %s\n", errStr);
        return 1;
    }
    printf("✓ Split BE partition: %u SMs\n", be_sms);

    // Step 4: 创建 Green Context
    CUdevResourceDesc sm_descs[2];
    CUgreenCtx green_ctxs[2];

    CHECK_CU(cuDevResourceGenerateDesc(&sm_descs[0], &sm_groups[0], 1));
    printf("✓ Generated HP resource descriptor\n");

    result = cuGreenCtxCreate(&green_ctxs[0], sm_descs[0], device, CU_GREEN_CTX_DEFAULT_STREAM);
    if (result != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(result, &errStr);
        printf("✗ cuGreenCtxCreate (HP) failed: %s\n", errStr);
        return 1;
    }
    printf("✓ Created HP Green Context\n");

    CHECK_CU(cuDevResourceGenerateDesc(&sm_descs[1], &sm_groups[1], 1));
    printf("✓ Generated BE resource descriptor\n");

    result = cuGreenCtxCreate(&green_ctxs[1], sm_descs[1], device, CU_GREEN_CTX_DEFAULT_STREAM);
    if (result != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(result, &errStr);
        printf("✗ cuGreenCtxCreate (BE) failed: %s\n", errStr);
        return 1;
    }
    printf("✓ Created BE Green Context\n");

    // Cleanup
    cuGreenCtxDestroy(green_ctxs[0]);
    cuGreenCtxDestroy(green_ctxs[1]);

    printf("\n✓ All Green Context API tests passed!\n");
    return 0;
}

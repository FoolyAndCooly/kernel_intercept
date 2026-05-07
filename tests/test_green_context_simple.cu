#include <cuda.h>
#include <stdio.h>

int main() {
    cuInit(0);
    CUdevice device;
    cuDeviceGet(&device, 0);
    
    // 获取 SM 数量
    int num_sms;
    cuDeviceGetAttribute(&num_sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
    printf("Total SMs: %d\n", num_sms);
    
    // 测试更小的分区
    CUdevResource full_sm_resource;
    cuDeviceGetDevResource(device, &full_sm_resource, CU_DEV_RESOURCE_TYPE_SM);
    
    // 尝试 56 + 56 (总共 112，小于 114)
    CUdevResource sm_groups[2];
    CUdevResource remaining;
    unsigned int hp_groups = 1;
    unsigned int hp_sms = 56;
    
    CUresult result = cuDevSmResourceSplitByCount(&sm_groups[0], &hp_groups, &full_sm_resource, &remaining, 0, hp_sms);
    if (result != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(result, &errStr);
        printf("HP split failed: %s\n", errStr);
        return 1;
    }
    printf("✓ HP split: %u SMs\n", hp_sms);
    
    unsigned int be_groups = 1;
    unsigned int be_sms = 56;
    result = cuDevSmResourceSplitByCount(&sm_groups[1], &be_groups, &remaining, NULL, 0, be_sms);
    if (result != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(result, &errStr);
        printf("BE split failed: %s\n", errStr);
        
        // 尝试更小的值
        be_sms = 48;
        result = cuDevSmResourceSplitByCount(&sm_groups[1], &be_groups, &remaining, NULL, 0, be_sms);
        if (result != CUDA_SUCCESS) {
            cuGetErrorString(result, &errStr);
            printf("BE split (48) also failed: %s\n", errStr);
            return 1;
        }
    }
    printf("✓ BE split: %u SMs\n", be_sms);
    
    printf("✓ Green Context partitioning works!\n");
    return 0;
}

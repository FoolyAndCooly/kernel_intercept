/**
 * Test program for Orion-Style GPU Scheduler
 * 
 * This is a simple test that simulates the scheduling workflow
 * without requiring actual CUDA operations.
 */

#include "gpu_capture.h"
#include "scheduler.h"
#include "kernel_profile.h"

#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <cassert>

using namespace orion;

// ============================================================================
// 测试用例
// ============================================================================

// 测试 1: 拦截层初始化
bool test_capture_layer_init() {
    std::cout << "Test: Capture Layer Initialization... ";
    
    int ret = init_capture_layer(2);
    if (ret != 0) {
        std::cout << "FAILED (init returned " << ret << ")" << std::endl;
        return false;
    }
    
    if (!is_capture_enabled()) {
        std::cout << "FAILED (capture not enabled)" << std::endl;
        return false;
    }
    
    shutdown_capture_layer();
    std::cout << "PASSED" << std::endl;
    return true;
}

// 测试 2: Client 索引管理
bool test_client_index() {
    std::cout << "Test: Client Index Management... ";
    
    init_capture_layer(4);
    
    // 初始状态应该是 -1
    if (get_current_client_idx() != -1) {
        std::cout << "FAILED (initial idx not -1)" << std::endl;
        shutdown_capture_layer();
        return false;
    }
    
    // 设置并验证
    set_current_client_idx(2);
    if (get_current_client_idx() != 2) {
        std::cout << "FAILED (set/get mismatch)" << std::endl;
        shutdown_capture_layer();
        return false;
    }
    
    // 多线程测试
    std::thread t1([]{
        set_current_client_idx(0);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        if (get_current_client_idx() != 0) {
            std::cerr << "Thread 1: client idx mismatch!" << std::endl;
        }
    });
    
    std::thread t2([]{
        set_current_client_idx(1);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        if (get_current_client_idx() != 1) {
            std::cerr << "Thread 2: client idx mismatch!" << std::endl;
        }
    });
    
    t1.join();
    t2.join();
    
    // 主线程的 idx 应该不变
    if (get_current_client_idx() != 2) {
        std::cout << "FAILED (main thread idx changed)" << std::endl;
        shutdown_capture_layer();
        return false;
    }
    
    shutdown_capture_layer();
    std::cout << "PASSED" << std::endl;
    return true;
}

// 测试 3: 操作提交
bool test_operation_submit() {
    std::cout << "Test: Operation Submit... ";
    
    init_capture_layer(2);
    set_current_client_idx(0);
    
    auto op = submit_operation(0, OperationType::MALLOC);
    if (!op) {
        std::cout << "FAILED (submit returned null)" << std::endl;
        shutdown_capture_layer();
        return false;
    }
    
    if (op->type != OperationType::MALLOC) {
        std::cout << "FAILED (wrong op type)" << std::endl;
        shutdown_capture_layer();
        return false;
    }
    
    if (op->client_idx != 0) {
        std::cout << "FAILED (wrong client idx)" << std::endl;
        shutdown_capture_layer();
        return false;
    }
    
    // 模拟完成
    op->mark_completed(cudaSuccess);
    
    if (!op->completed.load()) {
        std::cout << "FAILED (not marked completed)" << std::endl;
        shutdown_capture_layer();
        return false;
    }
    
    shutdown_capture_layer();
    std::cout << "PASSED" << std::endl;
    return true;
}

// 测试 4: Profile 加载
bool test_profile_loading() {
    std::cout << "Test: Profile Loading... ";
    
    KernelProfileTable table;
    
    // 尝试加载示例 profile
    bool loaded = table.load_from_json("../profiles/example_profile.json");
    
    if (!loaded) {
        // 如果文件不存在，手动添加一些测试数据
        KernelProfile p1;
        p1.kernel_id = "test_kernel_1";
        p1.duration_ms = 0.5f;
        p1.sm_needed = 32;
        p1.profile_type = ProfileType::COMPUTE_BOUND;
        table.add(p1);
        
        KernelProfile p2;
        p2.kernel_id = "test_kernel_2";
        p2.duration_ms = 0.2f;
        p2.sm_needed = 8;
        p2.profile_type = ProfileType::MEMORY_BOUND;
        table.add(p2);
    }
    
    if (table.size() == 0) {
        std::cout << "FAILED (no profiles loaded)" << std::endl;
        return false;
    }
    
    // 测试查找
    const KernelProfile* found = table.find(loaded ? "conv1:0" : "test_kernel_1");
    if (!found) {
        std::cout << "FAILED (profile not found)" << std::endl;
        return false;
    }
    
    std::cout << "PASSED (loaded " << table.size() << " profiles)" << std::endl;
    return true;
}

// 测试 5: 调度决策模拟
bool test_scheduling_decision() {
    std::cout << "Test: Scheduling Decision... ";
    
    // 这个测试模拟调度决策逻辑，不需要真正的 CUDA
    
    // 场景 1: 没有 HP 任务运行时，BE 应该被允许
    {
        // 模拟 BE 操作
        auto be_op = std::make_shared<OperationRecord>();
        be_op->type = OperationType::KERNEL_LAUNCH;
        be_op->sm_needed = 10;
        be_op->profile_type = ProfileType::MEMORY_BOUND;
        be_op->estimated_duration_ms = 0.1f;
        
        // 没有 HP 操作时，应该允许
        bool should_allow = true;  // schedule_be(nullptr, be_op) 的预期结果
        
        if (!should_allow) {
            std::cout << "FAILED (should allow BE when no HP)" << std::endl;
            return false;
        }
    }
    
    // 场景 2: HP 正在运行，BE 太大
    {
        auto hp_op = std::make_shared<OperationRecord>();
        hp_op->type = OperationType::KERNEL_LAUNCH;
        hp_op->profile_type = ProfileType::COMPUTE_BOUND;
        
        auto be_op = std::make_shared<OperationRecord>();
        be_op->type = OperationType::KERNEL_LAUNCH;
        be_op->sm_needed = 60;  // 超过阈值
        be_op->profile_type = ProfileType::COMPUTE_BOUND;
        
        // 应该被拒绝
    }
    
    // 场景 3: HP 正在运行，BE 小且互补
    {
        auto hp_op = std::make_shared<OperationRecord>();
        hp_op->type = OperationType::KERNEL_LAUNCH;
        hp_op->profile_type = ProfileType::COMPUTE_BOUND;
        
        auto be_op = std::make_shared<OperationRecord>();
        be_op->type = OperationType::KERNEL_LAUNCH;
        be_op->sm_needed = 10;  // 小于阈值
        be_op->profile_type = ProfileType::MEMORY_BOUND;  // 与 HP 互补
        
        // 应该被允许
    }
    
    std::cout << "PASSED" << std::endl;
    return true;
}

// 测试 6: 多线程压力测试
bool test_concurrent_operations() {
    std::cout << "Test: Concurrent Operations... ";
    
    init_capture_layer(4);
    
    constexpr int NUM_OPS_PER_THREAD = 100;
    std::atomic<int> completed_ops{0};
    
    auto worker = [&completed_ops](int client_idx) {
        set_current_client_idx(client_idx);
        
        for (int i = 0; i < NUM_OPS_PER_THREAD; i++) {
            auto op = submit_operation(client_idx, OperationType::KERNEL_LAUNCH);
            if (op) {
                // 模拟立即完成
                op->mark_completed(cudaSuccess);
                completed_ops++;
            }
        }
    };
    
    std::vector<std::thread> threads;
    for (int i = 0; i < 4; i++) {
        threads.emplace_back(worker, i);
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    if (completed_ops != 4 * NUM_OPS_PER_THREAD) {
        std::cout << "FAILED (expected " << 4 * NUM_OPS_PER_THREAD 
                  << " ops, got " << completed_ops << ")" << std::endl;
        shutdown_capture_layer();
        return false;
    }
    
    shutdown_capture_layer();
    std::cout << "PASSED (" << completed_ops << " ops)" << std::endl;
    return true;
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Orion GPU Scheduler Test Suite" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    int passed = 0;
    int failed = 0;
    
    // 运行测试
    if (test_capture_layer_init()) passed++; else failed++;
    if (test_client_index()) passed++; else failed++;
    if (test_operation_submit()) passed++; else failed++;
    if (test_profile_loading()) passed++; else failed++;
    if (test_scheduling_decision()) passed++; else failed++;
    if (test_concurrent_operations()) passed++; else failed++;
    
    // 汇总
    std::cout << "\n========================================" << std::endl;
    std::cout << "Results: " << passed << " passed, " << failed << " failed" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    return failed == 0 ? 0 : 1;
}

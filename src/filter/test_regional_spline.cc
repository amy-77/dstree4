#include "conformal.h"
#include <iostream>

int main() {
    std::cout << "开始测试区域优化样条回归..." << std::endl;
    
    // 调用测试函数
    upcite::RESPONSE result = upcite::test_optimized_regional_spline();
    
    if (result == upcite::SUCCESS) {
        std::cout << "测试成功完成！" << std::endl;
        return 0;
    } else {
        std::cout << "测试失败！" << std::endl;
        return 1;
    }
} 
#include "../api/cpp/Bolt.h"

int main() {
    Bolt* obj = new Bolt();
    const std::string storage_path = "./";
    const std::string model_name = "lenet";
    std::string bolt_path = "/data/bolt/model_zoo/onnx_models/lenet/lenet_f32.bolt";
    obj->load(bolt_path, "CPU_HIGH_PERFORMANCE");
    return 0;    
}

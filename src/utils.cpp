#include "eye_blink_detector/utils.h"

namespace eb_detector {
    template <typename T>
    void getParam(const YAML::Node &params, const std::string &param_name, T &variable, const T &default_value) {
        if (params[param_name]) {
            variable = params[param_name].as<T>();
        } else {
            variable = default_value;
        }
    }
}
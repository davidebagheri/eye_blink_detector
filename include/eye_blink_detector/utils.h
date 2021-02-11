#ifndef EYE_BLINK_DETECTOR_UTILS_H
#define EYE_BLINK_DETECTOR_UTILS_H

#include "yaml-cpp/yaml.h"
#include <iostream>

namespace eb_detector{
    template <typename T>
    void getParam(const YAML::Node& params, const std::string& param_name, T& variable, const T& default_value) {
        if (params[param_name]) {
            variable = params[param_name].as<T>();
        } else {
            variable = default_value;
        }
    }
}
#endif //EYE_BLINK_DETECTOR_UTILS_H

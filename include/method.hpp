#pragma once
#include <map>
#include <string>
#include <utility>

const std::map<std::pair<int, int>, float>& get_method(const std::string& method_name);
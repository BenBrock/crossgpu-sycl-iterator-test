#pragma once

#include <ranges>
#include <span>
#include <level_zero/ze_api.h>

namespace helper {

void check_devices(std::span<cl::sycl::device> devices) {
  std::cout << devices.size() << " devices" << std::endl;
  std::cout << "X = device can access peer" << std::endl;
  std::cout << "O = device cannot access peer" << std::endl;

  for (size_t i = 0; i < devices.size(); i++) {
    for (size_t j = 0; j < devices.size(); j++) {
      auto device_handle = cl::sycl::get_native<cl::sycl::backend::ext_oneapi_level_zero>(devices[i]);
      auto peer_device_handle = cl::sycl::get_native<cl::sycl::backend::ext_oneapi_level_zero>(devices[j]);

      ze_bool_t value;
      zeDeviceCanAccessPeer(device_handle, peer_device_handle, &value);

      if (value) {
        std::cout << "X";
      } else {
        std::cout << "O";
      }

    }
    std::cout << std::endl;
  }
}

template <typename Selector>
std::vector<cl::sycl::device> get_numa_devices(Selector&& selector) {
  namespace sycl = cl::sycl;

  std::vector<sycl::device> devices;

  sycl::platform p(std::forward<Selector>(selector));
  auto root_devices = p.get_devices();

  for (auto&& root_device : root_devices) {
    using namespace sycl::info;
    auto subdevices = root_device.create_sub_devices<
                        partition_property::partition_by_affinity_domain>(
                          partition_affinity_domain::numa);

    for (auto&& subdevice : subdevices) {
      devices.push_back(subdevice);
    }
  }

  return devices;
}

template <typename Selector>
std::vector<cl::sycl::device> get_devices(Selector&& selector) {
  namespace sycl = cl::sycl;

  sycl::platform p(std::forward<Selector>(selector));
  return p.get_devices();
}

template <typename Range>
void print_range(Range&& r, std::string label = "") {
  size_t indent = 1;

  if (label != "") {
    std::cout << "\"" << label << "\": ";
    indent += label.size() + 4;
  }

  std::string indent_whitespace(indent, ' ');

  std::cout << "[";
  size_t columns = 10;
  size_t count = 1;
  for (auto iter = r.begin(); iter != r.end(); ++iter) {
    std::cout << static_cast<std::ranges::range_value_t<Range>>(*iter);

    auto next = iter;
    ++next;
    if (next != r.end()) {
      std::cout << ", ";
      if (count % columns == 0) {
        std::cout << "\n" << indent_whitespace;
      }
    }
    ++count;
  }
  std::cout << "]" << std::endl;
}

} // end helper

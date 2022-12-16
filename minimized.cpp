#include <CL/sycl.hpp>
#include <cstdio>
#include "helper.hpp"

template <typename T>
struct smart_pointer {
  T* pointer;
};

void write_from_pointer(std::vector<smart_pointer<int>>& ptrs, size_t size, cl::sycl::device device) {
  namespace sycl = cl::sycl;
  for (smart_pointer<int> p : ptrs) {
    sycl::queue q(device);

    // Here we obtain the raw C pointer,
    // which is then copied to the GPU.
    // (Has expected behavior: writes to memory, both local and on other devices.)
    int* ptr = p.pointer;
    q.parallel_for(sycl::range<1>(size),
                   [=](auto&& id) {
                     ptr[id] = id;
                   }).wait();
  }
}

void write_from_smart_pointer(std::vector<smart_pointer<int>>& ptrs, size_t size, cl::sycl::device device) {
  namespace sycl = cl::sycl;
  for (smart_pointer<int> p : ptrs) {
    sycl::queue q(device);

    // Here the smart pointer is copied to the GPU.
    // We write to the pointer value inside the smart pointer.
    // (Does NOT have expected behavior: writes to local memory, does not write to other devices.)
    smart_pointer<int> ptr = p;
    q.parallel_for(sycl::range<1>(size),
                   [=](auto&& id) {
                     ptr.pointer[id] = id;
                   }).wait();
  }
}

void write_from_smart_pointer_and_pass_pointer(std::vector<smart_pointer<int>>& ptrs, size_t size, cl::sycl::device device) {
  namespace sycl = cl::sycl;
  for (smart_pointer<int> p : ptrs) {
    sycl::queue q(device);

    // Here the smart pointer is copied to the GPU.
    // We write to the pointer value inside the smart pointer.
    // (Does NOT have expected behavior: writes to local memory, does not write to other devices.)
    smart_pointer<int> ptr = p;
    int* dummy = p.pointer;
    q.parallel_for(sycl::range<1>(size),
                   [=](auto&& id) {
                     if (id > size) {
                       dummy[id] = id;
                     } else {
                       ptr.pointer[id] = id;
                     }
                   }).wait();
  }
}

void print_buffers(auto&& ptrs, std::size_t size) {
  printf("Reading buffers...\n");
  for (size_t i = 0; i < ptrs.size(); i++) {
    sycl::queue q;
    std::vector<int> values(size);
    auto ptr = ptrs[i].pointer;
    q.memcpy(values.data(), ptr, sizeof(int)*size).wait();
    printf("%lu'th buffer:\n", i);
    helper::print_range(values);
  }
}

void clear_buffers(auto&& ptrs, std::size_t size) {
  for (auto&& ptr : ptrs) {
    sycl::queue q;
    q.fill(ptr.pointer, int(0), size).wait();
  }
}

int main(int argc, char** argv) {
  namespace sycl = cl::sycl;

  // Get GPU devices.
  auto selector = sycl::gpu_selector();
  auto devices = helper::get_devices(selector);

  sycl::context context(devices);

  std::vector<smart_pointer<int>> ptrs;

  helper::check_devices(devices);

  size_t size = 100;

  printf("Allocate buffer on each GPU...\n");
  for (auto&& device : devices) {
    int* ptr = sycl::malloc_device<int>(size, device, context);
    ptrs.push_back(smart_pointer<int>{ptr});
  }

  size_t device = 0;

  // Test with raw pointers.

  printf("Write to buffers from device %lu using raw pointer...\n", device);
  write_from_pointer(ptrs, size, devices[device]);

  printf("Reading buffers...\n");
  print_buffers(ptrs, size);

  printf("Clearing buffers...\n");
  clear_buffers(ptrs, size);

  // Test with smart pointer

  printf("Write to buffers from device %lu using smart pointer...\n", device);
  write_from_smart_pointer(ptrs, size, devices[device]);

  printf("Reading buffers...\n");
  print_buffers(ptrs, size);

  printf("Clearing buffers...\n");
  clear_buffers(ptrs, size);

  // Test with smart pointer, but also send raw pointer

  printf("Write to buffers from device %lu using smart pointer (and sending regular pointer)...\n", device);
  write_from_smart_pointer_and_pass_pointer(ptrs, size, devices[device]);

  printf("Reading buffers...\n");
  print_buffers(ptrs, size);

  return 0;
}

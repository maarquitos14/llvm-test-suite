// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

//==--device_implicitly_copyable.cpp - SYCL implicit device copyable test --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <cassert>
#include <iostream>

#include <sycl/sycl.hpp>

struct ACopyable {
  int i;
  ACopyable() = default;
  ACopyable(int _i) : i(_i) {}
  ACopyable(const ACopyable &x) : i(x.i) {}
};

template <> struct sycl::is_device_copyable<ACopyable> : std::true_type {};

int main() {
  constexpr size_t arr_size = 5;
  constexpr int ref_val = 14;
  sycl::queue q;
  {
    std::pair<int, float> pair_arr[arr_size];
    std::pair<int, float> pair{ref_val, ref_val};
    std::pair<int, float> result_pair_arr[arr_size];
    std::pair<int, float> result_pair;

    for (auto i = 0; i < arr_size; i++) {
      pair_arr[i].first = i;
      pair_arr[i].second = i;
    }

    {
      sycl::buffer<std::pair<int, float>, 1> buf_pair_arr{
          result_pair_arr, sycl::range<1>(arr_size)};
      sycl::buffer<std::pair<int, float>, 1> buf_pair{&result_pair,
                                                      sycl::range<1>(1)};

      q.submit([&](sycl::handler &cgh) {
         auto acc_pair_arr =
             sycl::accessor{buf_pair_arr, cgh, sycl::read_write};
         auto acc_pair = sycl::accessor{buf_pair, cgh, sycl::read_write};
         cgh.single_task([=]() {
           for (auto i = 0; i < arr_size; i++) {
             acc_pair_arr[i] = pair_arr[i];
           }
           acc_pair[0] = pair;
         });
       }).wait_and_throw();
    }

    for (auto i = 0; i < arr_size; i++) {
      assert(result_pair_arr[i].first == i);
      assert(result_pair_arr[i].second == i);
    }
    assert(result_pair.first == ref_val && result_pair.second == ref_val);
  }

  {
    std::pair<ACopyable, float> pair_arr[arr_size];
    std::pair<ACopyable, float> pair{ACopyable(ref_val), ref_val};
    std::pair<ACopyable, float> result_pair_arr[arr_size];
    std::pair<ACopyable, float> result_pair;

    for (auto i = 0; i < arr_size; i++) {
      pair_arr[i].first = ACopyable(i);
      pair_arr[i].second = i;
    }

    {
      sycl::buffer<std::pair<ACopyable, float>, 1> buf_pair_arr{
          result_pair_arr, sycl::range<1>(arr_size)};
      sycl::buffer<std::pair<ACopyable, float>, 1> buf_pair{&result_pair,
                                                            sycl::range<1>(1)};

      q.submit([&](sycl::handler &cgh) {
         auto acc_pair_arr =
             sycl::accessor{buf_pair_arr, cgh, sycl::read_write};
         auto acc_pair = sycl::accessor{buf_pair, cgh, sycl::read_write};
         cgh.single_task([=]() {
           for (auto i = 0; i < arr_size; i++) {
             acc_pair_arr[i] = pair_arr[i];
           }
           acc_pair[0] = pair;
         });
       }).wait_and_throw();
    }

    for (auto i = 0; i < arr_size; i++) {
      assert(result_pair_arr[i].first.i == i);
      assert(result_pair_arr[i].second == i);
    }
    assert(result_pair.first.i == ref_val && result_pair.second == ref_val);
  }

  {
    std::tuple<int, float, bool> tuple_arr[arr_size];
    std::tuple<int, float, bool> tuple{ref_val, ref_val, true};
    std::tuple<int, float, bool> result_tuple_arr[arr_size];
    std::tuple<int, float, bool> result_tuple;

    for (auto i = 0; i < arr_size; i++) {
      auto &t = tuple_arr[i];
      std::get<0>(t) = i;
      std::get<1>(t) = i;
      std::get<2>(t) = true;
    }

    {
      sycl::buffer<std::tuple<int, float, bool>, 1> buf_tuple_arr{
          result_tuple_arr, sycl::range<1>(arr_size)};
      sycl::buffer<std::tuple<int, float, bool>, 1> buf_tuple{
          &result_tuple, sycl::range<1>(1)};

      q.submit([&](sycl::handler &cgh) {
         auto acc_tuple_arr =
             sycl::accessor{buf_tuple_arr, cgh, sycl::read_write};
         auto acc_tuple = sycl::accessor{buf_tuple, cgh, sycl::read_write};
         cgh.single_task([=]() {
           for (auto i = 0; i < arr_size; i++) {
             acc_tuple_arr[i] = tuple_arr[i];
           }
           acc_tuple[0] = tuple;
         });
       }).wait_and_throw();
    }

    for (auto i = 0; i < arr_size; i++) {
      auto t = result_tuple_arr[i];
      assert(std::get<0>(t) == i);
      assert(std::get<1>(t) == i);
      assert(std::get<2>(t) == true);
    }
    assert(std::get<0>(result_tuple) == ref_val);
    assert(std::get<1>(result_tuple) == ref_val);
    assert(std::get<2>(result_tuple) == true);
  }

  {
    std::tuple<ACopyable, float, bool> tuple_arr[arr_size];
    std::tuple<ACopyable, float, bool> tuple{ACopyable(ref_val), ref_val, true};
    std::tuple<ACopyable, float, bool> result_tuple_arr[arr_size];
    std::tuple<ACopyable, float, bool> result_tuple;

    for (auto i = 0; i < arr_size; i++) {
      auto &t = tuple_arr[i];
      std::get<0>(t) = ACopyable(i);
      std::get<1>(t) = i;
      std::get<2>(t) = true;
    }

    {
      sycl::buffer<std::tuple<ACopyable, float, bool>, 1> buf_tuple_arr{
          result_tuple_arr, sycl::range<1>(arr_size)};
      sycl::buffer<std::tuple<ACopyable, float, bool>, 1> buf_tuple{
          &result_tuple, sycl::range<1>(1)};

      q.submit([&](sycl::handler &cgh) {
         auto acc_tuple_arr =
             sycl::accessor{buf_tuple_arr, cgh, sycl::read_write};
         auto acc_tuple = sycl::accessor{buf_tuple, cgh, sycl::read_write};
         cgh.single_task([=]() {
           for (auto i = 0; i < arr_size; i++) {
             acc_tuple_arr[i] = tuple_arr[i];
           }
           acc_tuple[0] = tuple;
         });
       }).wait_and_throw();
    }

    for (auto i = 0; i < arr_size; i++) {
      auto t = result_tuple_arr[i];
      assert(std::get<0>(t).i == i);
      assert(std::get<1>(t) == i);
      assert(std::get<2>(t) == true);
    }
    assert(std::get<0>(result_tuple).i == ref_val);
    assert(std::get<1>(result_tuple) == ref_val);
    assert(std::get<2>(result_tuple) == true);
  }

  {
    std::variant<int, float, bool> variant_arr[arr_size];
    std::variant<int, float, bool> variant{14};
    std::variant<int, float, bool> result_variant_arr[arr_size];
    std::variant<int, float, bool> result_variant;

    constexpr int variant_size = 3;
    for (auto i = 0; i < arr_size; i++) {
      auto &v = variant_arr[i];
      auto index = i % variant_size;
      if (index == 0) {
        v = i;
      } else if (index == 1) {
        v = (float)i;
      } else {
        v = true;
      }
    }

    {
      sycl::buffer<std::variant<int, float, bool>, 1> buf_variant_arr{
          result_variant_arr, sycl::range<1>(arr_size)};
      sycl::buffer<std::variant<int, float, bool>, 1> buf_variant{
          &result_variant, sycl::range<1>(1)};

      q.submit([&](sycl::handler &cgh) {
         auto acc_variant_arr =
             sycl::accessor{buf_variant_arr, cgh, sycl::read_write};
         auto acc_variant = sycl::accessor{buf_variant, cgh, sycl::read_write};
         cgh.single_task([=]() {
           for (auto i = 0; i < arr_size; i++) {
             acc_variant_arr[i] = variant_arr[i];
           }
           acc_variant[0] = variant;
         });
       }).wait_and_throw();
    }

    for (auto i = 0; i < arr_size; i++) {
      auto v = result_variant_arr[i];
      auto index = i % variant_size;
      if (index == 0) {
        assert(std::get<0>(v) == i);
      } else if (index == 1) {
        assert(std::get<1>(v) == i);
      } else {
        assert(std::get<2>(v) == true);
      }
    }
    assert(std::get<0>(result_variant) == ref_val);
  }

  {
    std::variant<ACopyable> variant_arr[arr_size];
    std::variant<ACopyable> variant;
    std::variant<ACopyable> result_variant_arr[arr_size];
    std::variant<ACopyable> result_variant;
    q.submit([&](sycl::handler &cgh) {
       cgh.single_task([=]() {
         // std::variant with complex types relies on virtual functions, so
         // they cannot be used within sycl kernels
         auto size = sizeof(variant_arr[0]);
         size = sizeof(variant);
       });
     }).wait_and_throw();
  }

  {
    std::array<int, arr_size> arr_arr[arr_size];
    std::array<int, arr_size> arr;
    std::array<int, arr_size> result_arr_arr[arr_size];
    std::array<int, arr_size> result_arr;

    for (auto i = 0; i < arr_size; i++) {
      auto &a = arr_arr[i];
      for (auto j = 0; j < arr_size; j++) {
        a[j] = j;
      }
      arr[i] = i;
    }

    {
      sycl::buffer<std::array<int, arr_size>, 1> buf_arr_arr{
          result_arr_arr, sycl::range<1>(arr_size)};
      sycl::buffer<std::array<int, arr_size>, 1> buf_arr{&result_arr,
                                                         sycl::range<1>(1)};

      q.submit([&](sycl::handler &cgh) {
         auto acc_arr_arr = sycl::accessor{buf_arr_arr, cgh, sycl::read_write};
         auto acc_arr = sycl::accessor{buf_arr, cgh, sycl::read_write};
         cgh.single_task([=]() {
           for (auto i = 0; i < arr_size; i++) {
             acc_arr_arr[i] = arr_arr[i];
           }
           acc_arr[0] = arr;
         });
       }).wait_and_throw();
    }

    for (auto i = 0; i < arr_size; i++) {
      auto a = result_arr_arr[i];
      for (auto j = 0; j < arr_size; j++) {
        assert(a[j] == j);
      }
      assert(result_arr[i] == i);
    }
  }

  {
    std::array<ACopyable, arr_size> arr_arr[arr_size];
    std::array<ACopyable, arr_size> arr;
    std::array<ACopyable, arr_size> result_arr_arr[arr_size];
    std::array<ACopyable, arr_size> result_arr;

    for (auto i = 0; i < arr_size; i++) {
      auto &a = arr_arr[i];
      for (auto j = 0; j < arr_size; j++) {
        a[j] = ACopyable(j);
      }
      arr[i] = ACopyable(i);
    }

    {
      sycl::buffer<std::array<ACopyable, arr_size>, 1> buf_arr_arr{
          result_arr_arr, sycl::range<1>(arr_size)};
      sycl::buffer<std::array<ACopyable, arr_size>, 1> buf_arr{
          &result_arr, sycl::range<1>(1)};

      q.submit([&](sycl::handler &cgh) {
         auto acc_arr_arr = sycl::accessor{buf_arr_arr, cgh, sycl::read_write};
         auto acc_arr = sycl::accessor{buf_arr, cgh, sycl::read_write};
         cgh.single_task([=]() {
           for (auto i = 0; i < arr_size; i++) {
             acc_arr_arr[i] = arr_arr[i];
           }
           acc_arr[0] = arr;
         });
       }).wait_and_throw();
    }

    for (auto i = 0; i < arr_size; i++) {
      auto a = result_arr_arr[i];
      for (auto j = 0; j < arr_size; j++) {
        assert(a[j].i == j);
      }
      assert(result_arr[i].i == i);
    }
  }

  {
    std::optional<int> opt_arr[arr_size];
    std::optional<int> opt;
    std::optional<int> result_opt_arr[arr_size];
    std::optional<int> result_opt;

    for (auto i = 0; i < arr_size; i++) {
      opt_arr[i] = i;
    }
    opt = ref_val;

    {
      sycl::buffer<std::optional<int>, 1> buf_opt_arr{result_opt_arr,
                                                      sycl::range<1>(arr_size)};
      sycl::buffer<std::optional<int>, 1> buf_opt{&result_opt,
                                                  sycl::range<1>(1)};

      q.submit([&](sycl::handler &cgh) {
         auto acc_opt_arr = sycl::accessor{buf_opt_arr, cgh, sycl::read_write};
         auto acc_opt = sycl::accessor{buf_opt, cgh, sycl::read_write};
         cgh.single_task([=]() {
           for (auto i = 0; i < arr_size; i++) {
             acc_opt_arr[i] = opt_arr[i];
           }
           acc_opt[0] = opt;
         });
       }).wait_and_throw();
    }

    for (auto i = 0; i < arr_size; i++) {
      assert(result_opt_arr[i] == i);
    }

      assert(result_opt == ref_val);
  }

  {
    std::optional<ACopyable> opt_arr[arr_size];
    std::optional<ACopyable> opt;
    std::optional<ACopyable> result_opt_arr[arr_size];
    std::optional<ACopyable> result_opt;

    for (auto i = 0; i < arr_size; i++) {
      opt_arr[i] = ACopyable(i);
    }
    opt = ACopyable(ref_val);

    {
      sycl::buffer<std::optional<ACopyable>, 1> buf_opt_arr{
          result_opt_arr, sycl::range<1>(arr_size)};
      sycl::buffer<std::optional<ACopyable>, 1> buf_opt{&result_opt,
                                                        sycl::range<1>(1)};

      q.submit([&](sycl::handler &cgh) {
         auto acc_opt_arr = sycl::accessor{buf_opt_arr, cgh, sycl::read_write};
         auto acc_opt = sycl::accessor{buf_opt, cgh, sycl::read_write};
         cgh.single_task([=]() {
           for (auto i = 0; i < arr_size; i++) {
             acc_opt_arr[i] = opt_arr[i];
           }
           acc_opt[0] = opt;
         });
       }).wait_and_throw();
    }

    for (auto i = 0; i < arr_size; i++) {
      assert(result_opt_arr[i]->i == i);
    }

    assert(result_opt->i == ref_val);
  }

  {
    std::string strv_arr_val[arr_size];
    std::string strv_val{std::to_string(ref_val)};
    std::string_view strv_arr[arr_size];
    std::string_view strv{strv_val};
    std::string_view result_strv_arr[arr_size];
    std::string_view result_strv;

    for (auto i = 0; i < arr_size; i++) {
      strv_arr_val[i] = std::to_string(i);
      strv_arr[i] = std::string_view{strv_arr_val[i]};
    }

    {
      sycl::buffer<std::string_view, 1> buf_string_view_arr{
          result_strv_arr, sycl::range<1>(arr_size)};
      sycl::buffer<std::string_view, 1> buf_string_view{&result_strv,
                                                        sycl::range<1>(1)};

      q.submit([&](sycl::handler &cgh) {
         auto acc_string_view_arr =
             sycl::accessor{buf_string_view_arr, cgh, sycl::read_write};
         auto acc_string_view =
             sycl::accessor{buf_string_view, cgh, sycl::read_write};
         cgh.single_task([=]() {
           for (auto i = 0; i < arr_size; i++) {
             acc_string_view_arr[i] = strv_arr[i];
           }
           acc_string_view[0] = strv;
         });
       }).wait_and_throw();
    }

    for (auto i = 0; i < arr_size; i++) {
      assert(result_strv_arr[i] == std::to_string(i));
    }

    assert(result_strv == std::to_string(ref_val));
  }

#if __cpp_lib_span >= 202002
  {
    std::vector<int> v(arr_size);
    std::span<int> s{v.data(), arr_size};
    std::span<int> result_s{v.data(), arr_size};

    for (auto i = 0; i < arr_size; i++) {
      s[i] = i;
    }

    {
      std::buffer<std::span<int>, 1> buf_span{&result_s, std::range<1>(1)};

      q.submit([&](std::handler &cgh) {
         auto acc_span_arr = std::accessor{buf_span, cgh, std::read_write};
         cgh.single_task([=]() {
           for (auto i = 0; i < arr_size; i++) {
             acc_span_arr[0][i] = s[i];
           }
         });
       }).wait_and_throw();
    }

    for (auto i = 0; i < arr_size; i++) {
      assert(result_s[i] == i);
    }
  }
#endif

  {
    std::vector<int> v(arr_size);
    sycl::span<int> s{v.data(), arr_size};
    sycl::span<int> result_s{v.data(), arr_size};

    for (auto i = 0; i < arr_size; i++) {
      s[i] = i;
    }

    {
      sycl::buffer<sycl::span<int>, 1> buf_span{&result_s, sycl::range<1>(1)};

      q.submit([&](sycl::handler &cgh) {
         auto acc_span_arr = sycl::accessor{buf_span, cgh, sycl::read_write};
         cgh.single_task([=]() {
           for (auto i = 0; i < arr_size; i++) {
             acc_span_arr[0][i] = s[i];
           }
         });
       }).wait_and_throw();
    }

    for (auto i = 0; i < arr_size; i++) {
      assert(result_s[i] == i);
    }
  }

  std::cout << "Test passed" << std::endl;
}

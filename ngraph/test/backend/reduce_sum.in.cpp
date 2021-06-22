// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <random>
#include <string>

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/ndarray.hpp"
#include "util/random.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

static std::mt19937_64 random_generator;

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

// ----------------------- keep dims = false ----------------------- //

NGRAPH_TEST(${BACKEND_NAME}, reduce_sum_to_scalar)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto axes = make_shared<op::Constant>(element::i32, Shape{2}, vector<int32_t>{0, 1});
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceSum>(A, axes, false), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::f32, Shape{});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{10}), read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_sum_large_1d_to_scalar)
{
    Shape shape{1000000};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto axes = make_shared<op::Constant>(element::i32, Shape{}, 0);
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceSum>(A, axes, false), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    random_generator.seed(2);
    vector<float> v_a(1000000, 0);
    double r = 0;
    for (int i = 0; i < 1000000; i++)
    {
        v_a[i] = static_cast<float>(random_generator() % 255);
        r += static_cast<double>(v_a[i]);
    }
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, v_a);
    auto result = backend->create_tensor(element::f32, Shape{});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});

    EXPECT_TRUE(
        test::all_close_f(vector<float>{static_cast<float>(r)}, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_sum_matrix_columns)
{
    Shape shape_a{3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{2};
    auto axes = make_shared<op::Constant>(element::i32, Shape{}, 0);
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceSum>(A, axes, false), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result = backend->create_tensor(element::f32, shape_rt);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{9, 12}), read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_sum_matrix_6d)
{
    Shape shape_a{2, 6, 4, 5, 7, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{2, 4, 5, 3};
    auto axes = make_shared<op::Constant>(element::i32, Shape{2}, vector<int32_t>{1, 4});
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceSum>(A, axes, false), ParameterVector{A});

    auto backend_wrk = runtime::Backend::create("${BACKEND_NAME}");
    auto backend_ref = runtime::Backend::create("INTERPRETER");

    // Create some tensors for input/output
    auto a_wrk = backend_wrk->create_tensor(element::f32, shape_a);
    auto a_ref = backend_ref->create_tensor(element::f32, shape_a);
    auto result_wrk = backend_wrk->create_tensor(element::f32, shape_rt);
    auto result_ref = backend_ref->create_tensor(element::f32, shape_rt);

    vector<float> inp_data(shape_size<const Shape>(shape_a));
    iota(inp_data.begin(), inp_data.end(), 1.f);
    copy_data(a_wrk, inp_data);
    copy_data(a_ref, inp_data);

    auto handle_wrk = backend_wrk->compile(f);
    auto handle_ref = backend_ref->compile(f);
    handle_wrk->call_with_validate({result_wrk}, {a_wrk});
    handle_ref->call_with_validate({result_ref}, {a_ref});

    EXPECT_TRUE(test::all_close_f(read_vector<float>(result_ref), read_vector<float>(result_wrk)));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_sum_matrix_rows)
{
    Shape shape_a{3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3};
    auto axes = make_shared<op::Constant>(element::i32, Shape{}, 1);
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceSum>(A, axes, false), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result = backend->create_tensor(element::f32, shape_rt);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{3, 7, 11}), read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_sum_3d_to_matrix_most_sig)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3, 3};
    auto axes = make_shared<op::Constant>(element::i32, Shape{}, 0);
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceSum>(A, axes, false), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto result = backend->create_tensor(element::f32, shape_rt);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{1 + 10 + 19,
                                                 2 + 11 + 20,
                                                 3 + 12 + 21,
                                                 4 + 13 + 22,
                                                 5 + 14 + 23,
                                                 6 + 15 + 24,
                                                 7 + 16 + 25,
                                                 8 + 17 + 26,
                                                 9 + 18 + 27}),
                                  read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_sum_3d_to_matrix_least_sig)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3, 3};
    auto axes = make_shared<op::Constant>(element::i32, Shape{}, 2);
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceSum>(A, axes, false), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto result = backend->create_tensor(element::f32, shape_rt);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{1 + 2 + 3,
                                                 4 + 5 + 6,
                                                 7 + 8 + 9,
                                                 10 + 11 + 12,
                                                 13 + 14 + 15,
                                                 16 + 17 + 18,
                                                 19 + 20 + 21,
                                                 22 + 23 + 24,
                                                 25 + 26 + 27}),
                                  read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_sum_3d_to_vector)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3};
    auto axes = make_shared<op::Constant>(element::i32, Shape{2}, vector<int32_t>{0, 1});
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceSum>(A, axes, false), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto result = backend->create_tensor(element::f32, shape_rt);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{1 + 10 + 19 + 4 + 13 + 22 + 7 + 16 + 25,
                                                 2 + 11 + 20 + 5 + 14 + 23 + 8 + 17 + 26,
                                                 3 + 12 + 21 + 6 + 15 + 24 + 9 + 18 + 27}),
                                  read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_sum_3d_to_scalar)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{};
    auto axes = make_shared<op::Constant>(element::i32, Shape{3}, vector<int32_t>{0, 1, 2});
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceSum>(A, axes, false), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto result = backend->create_tensor(element::f32, shape_rt);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{1 + 10 + 19 + 4 + 13 + 22 + 7 + 16 + 25 + 2 + 11 + 20 + 5 + 14 + 23 + 8 +
                       17 + 26 + 3 + 12 + 21 + 6 + 15 + 24 + 9 + 18 + 27}),
        read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_sum_3d_to_scalar_int32)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    Shape shape_rt{};
    auto axes = make_shared<op::Constant>(element::i32, Shape{3}, vector<int32_t>{0, 1, 2});
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceSum>(A, axes, false), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape_a);
    copy_data(a, vector<int32_t>{0x40000001, 10, 19, 4,  13, 22, 7,  16, 25, 2,  11, 20, 5, 14,
                                 23,         8,  17, 26, 3,  12, 21, 6,  15, 24, 9,  18, 27});
    auto result = backend->create_tensor(element::i32, shape_rt);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((vector<int32_t>{0x40000001 + 10 + 19 + 4 + 13 + 22 + 7 + 16 + 25 + 2 + 11 + 20 + 5 +
                               14 + 23 + 8 + 17 + 26 + 3 + 12 + 21 + 6 + 15 + 24 + 9 + 18 + 27}),
              read_vector<int32_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_sum_5d_to_scalar)
{
    Shape shape_a{3, 3, 3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{};
    auto axes = make_shared<op::Constant>(element::i32, Shape{5}, vector<int32_t>{0, 1, 2, 3, 4});
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceSum>(A, axes, false), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, std::vector<float>(std::pow(3, 5), 1));
    auto result = backend->create_tensor(element::f32, shape_rt);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(std::vector<float>{243.}, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_sum_5d_to_scalar_int32)
{
    Shape shape_a{3, 3, 3, 3, 3};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    Shape shape_rt{};
    auto axes = make_shared<op::Constant>(element::i32, Shape{5}, vector<int32_t>{0, 1, 2, 3, 4});
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceSum>(A, axes, false), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape_a);
    copy_data(a, std::vector<int32_t>(std::pow(3, 5), 1));
    auto result = backend->create_tensor(element::i32, shape_rt);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ(std::vector<int32_t>{243}, read_vector<int32_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_sum_2d_to_scalar_int8)
{
    Shape shape_a{3, 3};
    auto A = make_shared<op::Parameter>(element::i8, shape_a);
    Shape shape_rt{};
    auto axes = make_shared<op::Constant>(element::i32, Shape{2}, vector<int32_t>{0, 1});
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceSum>(A, axes, false), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i8, shape_a);
    copy_data(a, std::vector<int8_t>{1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto result = backend->create_tensor(element::i8, shape_rt);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ(std::vector<int8_t>{45}, read_vector<int8_t>(result));
}

#if NGRAPH_INTERPRETER_ENABLE

#ifndef _WIN32
NGRAPH_TEST(${BACKEND_NAME}, reduce_sum_stable_acc)
{
    std::string backend_name = "${BACKEND_NAME}";
    if (backend_name == "INTERPRETER")
    {
        return;
    }
    Shape shape_a{10, 10, 10, 30};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);

    Shape shape_rt{10};
    auto axes = make_shared<op::Constant>(element::i32, Shape{3}, vector<int32_t>{1, 2, 3});
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceSum>(A, axes, false), ParameterVector{A});

    test::Uniform<float> rng(1000.0f, 1000.1f, 2112);
    vector<vector<float>> args;
    for (shared_ptr<op::Parameter> param : f->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_shape()));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }

    auto ref_func = clone_function(*f);
    auto bk_func = clone_function(*f);

    auto ref_results = execute(ref_func, args, "INTERPRETER");
    auto bk_results = execute(bk_func, args, "${BACKEND_NAME}");

    EXPECT_TRUE(
        test::all_close_f(ref_results.at(0), bk_results.at(0), DEFAULT_FLOAT_TOLERANCE_BITS + 1));
}
#endif

NGRAPH_TEST(${BACKEND_NAME}, reduce_sum_stable_simple_float)
{
    std::string backend_name = "${BACKEND_NAME}";
    if (backend_name == "INTERPRETER")
    {
        return;
    }
    Shape shape_a{20};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);

    Shape shape_rt{};
    auto axes = make_shared<op::Constant>(element::i32, Shape{}, 0);
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceSum>(A, axes, false), ParameterVector{A});

    vector<vector<float>> args;
    args.push_back(vector<float>{10000000.0f, 0.9f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f,
                                 0.8f,        0.1f, 0.9f, 0.5f, 0.2f, 0.3f, 0.4f,
                                 0.5f,        0.6f, 0.7f, 0.8f, 0.9f, 0.1f});

    auto ref_func = clone_function(*f);
    auto bk_func = clone_function(*f);

    auto ref_results = execute(ref_func, args, "INTERPRETER");
    auto bk_results = execute(bk_func, args, "${BACKEND_NAME}");

    EXPECT_TRUE(
        test::all_close_f(ref_results.at(0), bk_results.at(0), DEFAULT_FLOAT_TOLERANCE_BITS - 1));
}

#endif

NGRAPH_TEST(${BACKEND_NAME}, reduce_sum_inf)
{
    Shape shape{7, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto axes = make_shared<op::Constant>(element::i32, Shape{}, 1);
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceSum>(A, axes, false), ParameterVector{A});

    auto infi = std::numeric_limits<float>::infinity();

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a,
              test::NDArray<float, 2>({{-infi, 0, 0, infi},
                                       {infi, 100, -100, -infi},
                                       {infi, 0, 100, infi},
                                       {-infi, -100, 0, -infi},
                                       {infi, infi, infi, infi},
                                       {infi, infi, infi, -infi},
                                       {infi, std::nanf(""), 42, infi}})
                  .get_vector());
    auto result = backend->create_tensor(element::f32, Shape{7});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    auto r = read_vector<float>(result);
    ASSERT_EQ(r.size(), 7);
    EXPECT_TRUE(isnan(r[0]));
    EXPECT_TRUE(isnan(r[1]));
    EXPECT_TRUE(r[2] > 0 && isinf(r[2]));
    EXPECT_TRUE(r[3] < 0 && isinf(r[3]));
    EXPECT_TRUE(r[4] > 0 && isinf(r[4]));
    EXPECT_TRUE(isnan(r[5]));
    EXPECT_TRUE(isnan(r[6]));
}

// ----------------------- keep dims = true ----------------------- //

NGRAPH_TEST(${BACKEND_NAME}, reduce_sum_keep_to_scalar)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto axes = make_shared<op::Constant>(element::i32, Shape{2}, vector<int32_t>{0, 1});
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceSum>(A, axes, true), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::f32, Shape{1, 1});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{10}), read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_sum_keep_large_1d_to_scalar)
{
    Shape shape{1000000};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto axes = make_shared<op::Constant>(element::i32, Shape{}, 0);
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceSum>(A, axes, true), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    random_generator.seed(2);
    vector<float> v_a(1000000, 0);
    double r = 0;
    for (int i = 0; i < 1000000; i++)
    {
        v_a[i] = static_cast<float>(random_generator() % 255);
        r += static_cast<double>(v_a[i]);
    }
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, v_a);
    auto result = backend->create_tensor(element::f32, Shape{1});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});

    EXPECT_TRUE(
        test::all_close_f(vector<float>{static_cast<float>(r)}, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_sum_keep_matrix_columns)
{
    Shape shape_a{3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{1, 2};
    auto axes = make_shared<op::Constant>(element::i32, Shape{}, 0);
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceSum>(A, axes, true), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result = backend->create_tensor(element::f32, shape_rt);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{9, 12}), read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_sum_keep_matrix_6d)
{
    Shape shape_a{2, 6, 4, 5, 7, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{2, 1, 4, 5, 1, 3};
    auto axes = make_shared<op::Constant>(element::i32, Shape{2}, vector<int32_t>{1, 4});
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceSum>(A, axes, true), ParameterVector{A});

    auto backend_wrk = runtime::Backend::create("${BACKEND_NAME}");
    auto backend_ref = runtime::Backend::create("INTERPRETER");

    // Create some tensors for input/output
    auto a_wrk = backend_wrk->create_tensor(element::f32, shape_a);
    auto a_ref = backend_ref->create_tensor(element::f32, shape_a);
    auto result_wrk = backend_wrk->create_tensor(element::f32, shape_rt);
    auto result_ref = backend_ref->create_tensor(element::f32, shape_rt);

    vector<float> inp_data(shape_size<const Shape>(shape_a));
    iota(inp_data.begin(), inp_data.end(), 1.f);
    copy_data(a_wrk, inp_data);
    copy_data(a_ref, inp_data);

    auto handle_wrk = backend_wrk->compile(f);
    auto handle_ref = backend_ref->compile(f);
    handle_wrk->call_with_validate({result_wrk}, {a_wrk});
    handle_ref->call_with_validate({result_ref}, {a_ref});

    EXPECT_TRUE(test::all_close_f(read_vector<float>(result_ref), read_vector<float>(result_wrk)));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_sum_keep_matrix_rows)
{
    Shape shape_a{3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3, 1};
    auto axes = make_shared<op::Constant>(element::i32, Shape{}, 1);
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceSum>(A, axes, true), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result = backend->create_tensor(element::f32, shape_rt);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{3, 7, 11}), read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_sum_keep_3d_to_matrix_most_sig)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{1, 3, 3};
    auto axes = make_shared<op::Constant>(element::i32, Shape{}, 0);
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceSum>(A, axes, true), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto result = backend->create_tensor(element::f32, shape_rt);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{1 + 10 + 19,
                                                 2 + 11 + 20,
                                                 3 + 12 + 21,
                                                 4 + 13 + 22,
                                                 5 + 14 + 23,
                                                 6 + 15 + 24,
                                                 7 + 16 + 25,
                                                 8 + 17 + 26,
                                                 9 + 18 + 27}),
                                  read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_sum_keep_3d_to_matrix_least_sig)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3, 3, 1};
    auto axes = make_shared<op::Constant>(element::i32, Shape{}, 2);
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceSum>(A, axes, true), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto result = backend->create_tensor(element::f32, shape_rt);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{1 + 2 + 3,
                                                 4 + 5 + 6,
                                                 7 + 8 + 9,
                                                 10 + 11 + 12,
                                                 13 + 14 + 15,
                                                 16 + 17 + 18,
                                                 19 + 20 + 21,
                                                 22 + 23 + 24,
                                                 25 + 26 + 27}),
                                  read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_sum_keep_3d_to_vector)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{1, 1, 3};
    auto axes = make_shared<op::Constant>(element::i32, Shape{2}, vector<int32_t>{0, 1});
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceSum>(A, axes, true), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto result = backend->create_tensor(element::f32, shape_rt);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{1 + 10 + 19 + 4 + 13 + 22 + 7 + 16 + 25,
                                                 2 + 11 + 20 + 5 + 14 + 23 + 8 + 17 + 26,
                                                 3 + 12 + 21 + 6 + 15 + 24 + 9 + 18 + 27}),
                                  read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_sum_keep_3d_to_scalar)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{1, 1, 1};
    auto axes = make_shared<op::Constant>(element::i32, Shape{3}, vector<int32_t>{0, 1, 2});
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceSum>(A, axes, true), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto result = backend->create_tensor(element::f32, shape_rt);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{1 + 10 + 19 + 4 + 13 + 22 + 7 + 16 + 25 + 2 + 11 + 20 + 5 + 14 + 23 + 8 +
                       17 + 26 + 3 + 12 + 21 + 6 + 15 + 24 + 9 + 18 + 27}),
        read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_sum_keep_3d_to_scalar_int32)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    Shape shape_rt{1, 1, 1};
    auto axes = make_shared<op::Constant>(element::i32, Shape{3}, vector<int32_t>{0, 1, 2});
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceSum>(A, axes, true), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape_a);
    copy_data(a, vector<int32_t>{0x40000001, 10, 19, 4,  13, 22, 7,  16, 25, 2,  11, 20, 5, 14,
                                 23,         8,  17, 26, 3,  12, 21, 6,  15, 24, 9,  18, 27});
    auto result = backend->create_tensor(element::i32, shape_rt);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((vector<int32_t>{0x40000001 + 10 + 19 + 4 + 13 + 22 + 7 + 16 + 25 + 2 + 11 + 20 + 5 +
                               14 + 23 + 8 + 17 + 26 + 3 + 12 + 21 + 6 + 15 + 24 + 9 + 18 + 27}),
              read_vector<int32_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_sum_keep_5d_to_scalar)
{
    Shape shape_a{3, 3, 3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{1, 1, 1, 1, 1};
    auto axes = make_shared<op::Constant>(element::i32, Shape{5}, vector<int32_t>{0, 1, 2, 3, 4});
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceSum>(A, axes, true), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, std::vector<float>(std::pow(3, 5), 1));
    auto result = backend->create_tensor(element::f32, shape_rt);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(std::vector<float>{243.}, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_sum_keep_5d_to_scalar_int32)
{
    Shape shape_a{3, 3, 3, 3, 3};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    Shape shape_rt{1, 1, 1, 1, 1};
    auto axes = make_shared<op::Constant>(element::i32, Shape{5}, vector<int32_t>{0, 1, 2, 3, 4});
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceSum>(A, axes, true), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape_a);
    copy_data(a, std::vector<int32_t>(std::pow(3, 5), 1));
    auto result = backend->create_tensor(element::i32, shape_rt);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ(std::vector<int32_t>{243}, read_vector<int32_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_sum_keep_2d_to_scalar_int8)
{
    Shape shape_a{3, 3};
    auto A = make_shared<op::Parameter>(element::i8, shape_a);
    Shape shape_rt{1, 1};
    auto axes = make_shared<op::Constant>(element::i32, Shape{2}, vector<int32_t>{0, 1});
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceSum>(A, axes, true), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i8, shape_a);
    copy_data(a, std::vector<int8_t>{1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto result = backend->create_tensor(element::i8, shape_rt);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ(std::vector<int8_t>{45}, read_vector<int8_t>(result));
}

#if NGRAPH_INTERPRETER_ENABLE

#ifndef _WIN32
NGRAPH_TEST(${BACKEND_NAME}, reduce_sum_keep_stable_acc)
{
    std::string backend_name = "${BACKEND_NAME}";
    if (backend_name == "INTERPRETER")
    {
        return;
    }
    Shape shape_a{10, 10, 10, 30};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);

    Shape shape_rt{10, 1, 1, 1};
    auto axes = make_shared<op::Constant>(element::i32, Shape{3}, vector<int32_t>{1, 2, 3});
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceSum>(A, axes, true), ParameterVector{A});

    test::Uniform<float> rng(1000.0f, 1000.1f, 2112);
    vector<vector<float>> args;
    for (shared_ptr<op::Parameter> param : f->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_shape()));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }

    auto ref_func = clone_function(*f);
    auto bk_func = clone_function(*f);

    auto ref_results = execute(ref_func, args, "INTERPRETER");
    auto bk_results = execute(bk_func, args, "${BACKEND_NAME}");

    EXPECT_TRUE(
        test::all_close_f(ref_results.at(0), bk_results.at(0), DEFAULT_FLOAT_TOLERANCE_BITS + 1));
}
#endif


NGRAPH_TEST(${BACKEND_NAME}, reduce_sum_keep_stable_simple_float)
{
    std::string backend_name = "${BACKEND_NAME}";
    if (backend_name == "INTERPRETER")
    {
        return;
    }
    Shape shape_a{20};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);

    Shape shape_rt{1};
    auto axes = make_shared<op::Constant>(element::i32, Shape{}, 0);
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceSum>(A, axes, true), ParameterVector{A});

    vector<vector<float>> args;
    args.push_back(vector<float>{10000000.0f, 0.9f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f,
                                 0.8f,        0.1f, 0.9f, 0.5f, 0.2f, 0.3f, 0.4f,
                                 0.5f,        0.6f, 0.7f, 0.8f, 0.9f, 0.1f});

    auto ref_func = clone_function(*f);
    auto bk_func = clone_function(*f);

    auto ref_results = execute(ref_func, args, "INTERPRETER");
    auto bk_results = execute(bk_func, args, "${BACKEND_NAME}");

    EXPECT_TRUE(
        test::all_close_f(ref_results.at(0), bk_results.at(0), DEFAULT_FLOAT_TOLERANCE_BITS - 1));
}

#endif

NGRAPH_TEST(${BACKEND_NAME}, reduce_sum_keep_inf)
{
    Shape shape{7, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto axes = make_shared<op::Constant>(element::i32, Shape{}, 1);
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceSum>(A, axes, true), ParameterVector{A});

    auto infi = std::numeric_limits<float>::infinity();

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a,
              test::NDArray<float, 2>({{-infi, 0, 0, infi},
                                       {infi, 100, -100, -infi},
                                       {infi, 0, 100, infi},
                                       {-infi, -100, 0, -infi},
                                       {infi, infi, infi, infi},
                                       {infi, infi, infi, -infi},
                                       {infi, std::nanf(""), 42, infi}})
                  .get_vector());
    auto result = backend->create_tensor(element::f32, Shape{7, 1});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    auto r = read_vector<float>(result);
    ASSERT_EQ(r.size(), 7);
    EXPECT_TRUE(isnan(r[0]));
    EXPECT_TRUE(isnan(r[1]));
    EXPECT_TRUE(r[2] > 0 && isinf(r[2]));
    EXPECT_TRUE(r[3] < 0 && isinf(r[3]));
    EXPECT_TRUE(r[4] > 0 && isinf(r[4]));
    EXPECT_TRUE(isnan(r[5]));
    EXPECT_TRUE(isnan(r[6]));
}

// Dynamic

NGRAPH_TEST(${BACKEND_NAME}, reduce_sum_matrix_columns_dynamic)
{
    auto A = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto axes = make_shared<op::Constant>(element::i32, Shape{}, 0);
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceSum>(A, axes, false), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}", true);

    // Create some tensors for input/output
    Shape shape_a{3, 2};
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{9, 12}), read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_sum_matrix_rows_dynamic)
{
    auto A = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto axes = make_shared<op::Constant>(element::i32, Shape{}, 1);
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceSum>(A, axes, false), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}", true);

    // Create some tensors for input/output
    Shape shape_a{3, 2};
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{3, 7, 11}), read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_sum_keep_matrix_columns_dynamic)
{
    auto A = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto axes = make_shared<op::Constant>(element::i32, Shape{}, 0);
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceSum>(A, axes, true), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}", true);

    // Create some tensors for input/output
    Shape shape_a{3, 2};
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{9, 12}), read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_sum_keep_matrix_rows_dynamic)
{
    auto A = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto axes = make_shared<op::Constant>(element::i32, Shape{}, 1);
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceSum>(A, axes, true), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}", true);

    // Create some tensors for input/output
    Shape shape_a{3, 2};
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{3, 7, 11}), read_vector<float>(result)));
}

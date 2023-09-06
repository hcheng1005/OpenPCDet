/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */
 
#ifndef __SPCONV_ENGINE_HPP__
#define __SPCONV_ENGINE_HPP__

#include <memory>
#include <string>
#include <vector>
#include <spconv/tensor.hpp>

namespace spconv {

#define Exported __attribute__((visibility("default")))

enum class Precision : int { None = 0, Float16 = 1, Int8 = 2 };

/**
  Storage of data tensor
**/
class SparseDTensor {
 public:
  virtual Tensor features() const = 0;
  virtual Tensor indices() const  = 0;

  virtual std::vector<int> grid_size() const = 0;
  virtual int device() const = 0;

  virtual std::string name() const = 0;

  virtual void set_data(
    const std::vector<int64_t>& features_shape,
    DataType features_dtype, void* features_data,
    const std::vector<int64_t>& indices_shape, DataType indices_dtype,
    void* indices_data, std::vector<int> grid_size
  ) = 0;
};

/**
  Engine types for sparse convolution
**/
class Engine {
 public:
  /**
    Inference function for sparse convolution

    features_shape: The shape of the input feature matrix, it must be two elements.
    features_dtype: The data type of the input feature matrix, it must be Float16 now.
    features_data:  The data pointer of the input feature matrix
    indices_shape:  The shape of the input indices matrix, it must be two elements[n, 4]
    indices_dtype:  The data type of the input indices matrix, it must be Int32 now.
    indices_data:   The data pointer of the input indices matrix
    batch:          The batch size of the input, it must be 1 now.
    grid_size:      The grid size of the input data, For example: 41,1440,1440 or 1440,1440,41
    stream:         Which stream is expected to enqueue the inference.
  **/
  Exported virtual void forward(void* stream = nullptr) = 0;
  Exported virtual size_t num_input() const = 0;
  Exported virtual SparseDTensor* input(unsigned int index) = 0;
  Exported virtual size_t num_output() const = 0;
  Exported virtual SparseDTensor* output(unsigned int index) = 0;

  // If you change the precision of a node after loading the model, you should call this function to
  // reconfigure it
  Exported virtual void reconfigure(void* stream = nullptr) = 0;

  // If you want to execute an implicit PTQ calibration, you can enable int8calibration by marking
  // it and collecting the maximum value of the tensor in the next forward.
  Exported virtual void set_int8_calibration(bool enable, void* stream = nullptr) = 0;

  // You can modify the precision of a node with this function, but don't forget to call reconfigure
  Exported virtual void set_node_precision_byname(const char* name, Precision compute_precision,
                                                  Precision output_precision) = 0;
  Exported virtual void set_node_precision_byoptype(const char* optype, Precision compute_precision,
                                                    Precision output_precision) = 0;
};

class ITensor{
public:
  virtual std::string name() = 0;
};

class INode{
public:
  virtual std::string name() = 0;
  virtual std::string optype() = 0;
  virtual ITensor* input(unsigned int index) = 0;
  virtual ITensor* output(unsigned int index) = 0;

  virtual unsigned int num_output() = 0;
  virtual unsigned int num_input() = 0;
};

class EngineBuilder{
public:
  Exported virtual ITensor* push_input(const std::string& name) = 0;
  Exported virtual INode* push_add(
      const std::string& name, 
      ITensor* a, 
      ITensor* b,
      float a_dynamic_range,
      float b_dynamic_range,
      const std::string& output_name,
      Precision precision, Precision output_precision) = 0;

  Exported virtual INode* push_relu(
      const std::string& name, 
      ITensor* x, 
      const std::string& output_name) = 0;

  Exported virtual INode* push_dense(
      const std::string& name, ITensor* x,
      const std::string& format,
      const std::string& output_name,
      const std::vector<int>& input_spatial_shape,
      const std::vector<int>& output_shape) = 0;

  Exported virtual INode* push_reshape(
      const std::string& name, ITensor* x, 
      const std::vector<int64_t>& shape,
      const std::string& output_name) = 0;

  Exported virtual INode* push_transpose(
      const std::string& name, ITensor* x, 
      const std::vector<int64_t>& dims,
      const std::string& output_name) = 0;

  Exported virtual INode* push_sparse_conv(
      const std::string& name, 
      ITensor* x,
      const std::vector<unsigned short>& weight,
      const std::vector<int>& weight_shape,
      const std::vector<float>& weight_dynamic_ranges,
      const std::vector<unsigned short>& bias,
      const std::vector<int>& bias_shape,
      const std::string& activation,
      const std::vector<int>& kernel_size,
      const std::vector<int>& stride,
      const std::vector<int>& padding,
      const std::vector<int>& dilation,
      float input_dynamic_range,
      bool submanifold,
      int max_output_points,
      const std::string& rulebook,
      Precision precision,
      Precision output_precision,
      const std::string& output_name) = 0;

  Exported virtual void push_output(ITensor* value) = 0;

  // build engine
  Exported virtual std::shared_ptr<Engine> build(Precision precision, void* stream = nullptr) = 0;
};

/**
 * To build a engine.
*/
Exported std::shared_ptr<EngineBuilder> create_engine_builder();

/**
  Enable detailed information output

  enable: You should set this to true if you want to debug the model inference process. default:
  false
*/
Exported void set_verbose(bool enable);
Exported const char* get_precision_string(Precision precision);

};  // namespace spconv

#endif  // #ifndef __SPCONV_ENGINE_HPP__
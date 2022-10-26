# Integrate OpenVINO™ with Your Application {#openvino_docs_OV_UG_Integrate_OV_with_your_application}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_docs_OV_UG_Model_Representation
   openvino_docs_OV_UG_Infer_request
   openvino_docs_OV_UG_Python_API_exclusives

@endsphinxdirective


Following these steps, you can implement a typical OpenVINO™ Runtime inference pipeline in your application. Before proceeding, make sure you have [installed OpenVINO Runtime](../install_guides/installing-openvino-runtime.md) and set environment variables (run `<INSTALL_DIR>/setupvars.sh` for Linux or `setupvars.bat` for Windows, otherwise, the `OpenVINO_DIR` variable won't be configured properly to pass `find_package` calls).


![ie_api_use_cpp]

## Step 1. Create OpenVINO™ Runtime Core 

Include next files to work with OpenVINO™ Runtime:

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/src/main.cpp include

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/src/main.py import

@endsphinxtab

@sphinxtab{C}

@snippet docs/snippets/src/main.c include

@endsphinxtab

@endsphinxtabset

Use the following code to create OpenVINO™ Core to manage available devices and read model objects:

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/src/main.cpp part1

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/src/main.py part1

@endsphinxtab

@sphinxtab{C}

@snippet docs/snippets/src/main.c part1

@endsphinxtab

@endsphinxtabset

## Step 2. Compile the Model

`ov::CompiledModel` class represents a device specific compiled model. `ov::CompiledModel` allows you to get information inputs or output ports by a tensor name or index. This approach is aligned with the majority of frameworks.

Compile the model for a specific device using `ov::Core::compile_model()`:

@sphinxtabset

@sphinxtab{C++}

@sphinxtabset

@sphinxtab{IR}

@snippet docs/snippets/src/main.cpp part2_1

@endsphinxtab

@sphinxtab{ONNX}

@snippet docs/snippets/src/main.cpp part2_2

@endsphinxtab

@sphinxtab{PaddlePaddle}

@snippet docs/snippets/src/main.cpp part2_3

@endsphinxtab

@sphinxtab{ov::Model}

@snippet docs/snippets/src/main.cpp part2_4

@endsphinxtab

@endsphinxtabset

@endsphinxtab

@sphinxtab{Python}

@sphinxtabset

@sphinxtab{IR}

@snippet docs/snippets/src/main.py part2_1

@endsphinxtab

@sphinxtab{ONNX}

@snippet docs/snippets/src/main.py part2_2

@endsphinxtab

@sphinxtab{PaddlePaddle}

@snippet docs/snippets/src/main.py part2_3

@endsphinxtab

@sphinxtab{ov::Model}

@snippet docs/snippets/src/main.py part2_4

@endsphinxtab

@endsphinxtabset

@endsphinxtab

@sphinxtab{C}

@sphinxtabset

@sphinxtab{IR}

@snippet docs/snippets/src/main.c part2_1

@endsphinxtab

@sphinxtab{ONNX}

@snippet docs/snippets/src/main.c part2_2

@endsphinxtab

@sphinxtab{PaddlePaddle}

@snippet docs/snippets/src/main.c part2_3

@endsphinxtab

@sphinxtab{ov::Model}

@snippet docs/snippets/src/main.c part2_4

@endsphinxtab

@endsphinxtabset

@endsphinxtab

@endsphinxtabset

The `ov::Model` object represents any models inside the OpenVINO™ Runtime.
For more details please read article about [OpenVINO™ Model representation](model_representation.md).

The code above creates a compiled model associated with a single hardware device from the model object.
It is possible to create as many compiled models as needed and use them simultaneously (up to the limitation of the hardware resources).
To learn how to change the device configuration, read the [Query device properties](./supported_plugins/config_properties.md) article.

## Step 3. Create an Inference Request

`ov::InferRequest` class provides methods for model inference in OpenVINO™ Runtime. Create an infer request using the following code (see [InferRequest detailed documentation](./ov_infer_request.md) for more details):

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/src/main.cpp part3

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/src/main.py part3

@endsphinxtab

@sphinxtab{C}

@snippet docs/snippets/src/main.c part3

@endsphinxtab

@endsphinxtabset

## Step 4. Set Inputs

You can use external memory to create `ov::Tensor` and use the `ov::InferRequest::set_input_tensor` method to put this tensor on the device:

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/src/main.cpp part4

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/src/main.py part4

@endsphinxtab

@sphinxtab{C}

@snippet docs/snippets/src/main.c part4

@endsphinxtab

@endsphinxtabset

## Step 5. Start Inference

OpenVINO™ Runtime supports inference in either synchronous or asynchronous mode. Using the Async API can improve application's overall frame-rate: instead of waiting for inference to complete, the app can keep working on the host while the accelerator is busy. You can use `ov::InferRequest::start_async` to start model inference in the asynchronous mode and call `ov::InferRequest::wait` to wait for the inference results:

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/src/main.cpp part5

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/src/main.py part5

@endsphinxtab

@sphinxtab{C}

@snippet docs/snippets/src/main.c part5

@endsphinxtab

@endsphinxtabset

This section demonstrates a simple pipeline. To get more information about other ways to perform inference, read the dedicated ["Run inference" section](./ov_infer_request.md).

## Step 6. Process the Inference Results 

Go over the output tensors and process the inference results.

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/src/main.cpp part6

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/src/main.py part6

@endsphinxtab

@sphinxtab{C}

@snippet docs/snippets/src/main.c part6

@endsphinxtab

@endsphinxtabset

## Step 7. Release the allocated objects (only for C)

To avoid memory leak, applications developed with C API need to release the allocated objects in order.

@sphinxtabset

@sphinxtab{C}

@snippet docs/snippets/src/main.c part8

@endsphinxtab

@endsphinxtabset

## Step 8. Link and Build Your Application with OpenVINO™ Runtime (example)

This step may differ for different projects. In this example, a C++ & C application is used, together with CMake for project configuration.

### Create Structure for project:

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/src/main.cpp part7

@endsphinxtab

@sphinxtab{C}

@snippet docs/snippets/src/main.c part7

@endsphinxtab

@endsphinxtabset

### Create Cmake Script

For details on additional CMake build options, refer to the [CMake page](https://cmake.org/cmake/help/latest/manual/cmake.1.html#manual:cmake(1)).

@sphinxtabset

@sphinxtab{C++}

@snippet snippets/CMakeLists.txt cmake:integration_example_cpp

@endsphinxtab

@sphinxtab{C}

@snippet snippets/CMakeLists.txt cmake:integration_example_c

@endsphinxtab

@endsphinxtabset


### Build Project

To build your project using CMake with the default build tools currently available on your machine, execute the following commands:

```sh
cd build/
cmake ../project
cmake --build .
```

## Additional Resources

 - See the [OpenVINO Samples](Samples_Overview.md) page or the [Open Model Zoo Demos](https://docs.openvino.ai/nightly/omz_demos.html) page for specific examples of how OpenVINO pipelines are implemented for applications like image classification, text prediction, and many others.
 - [OpenVINO™ Runtime Preprocessing](./preprocessing_overview.md)
 - [Using Encrypted Models with OpenVINO&trade;](./protecting_model_guide.md)
 - [Open Model Zoo Demos](https://docs.openvino.ai/nightly/omz_demos.html)


[ie_api_flow_cpp]: img/BASIC_IE_API_workflow_Cpp.svg
[ie_api_use_cpp]: img/IMPLEMENT_PIPELINE_with_API_C.svg
[ie_api_flow_python]: img/BASIC_IE_API_workflow_Python.svg
[ie_api_use_python]: img/IMPLEMENT_PIPELINE_with_API_Python.svg

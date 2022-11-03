# Converting a Kaldi Model {#openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_Kaldi}

> **NOTE**: Model Optimizer supports the [nnet1](http://kaldi-asr.org/doc/dnn1.html) and [nnet2](http://kaldi-asr.org/doc/dnn2.html) formats of Kaldi models. The support of the [nnet3](http://kaldi-asr.org/doc/dnn3.html) format is limited.
 
<a name="Convert_From_Kaldi"></a>To convert a Kaldi model, run Model Optimizer with the path to the input model `.nnet` or `.mdl` file:

```sh
 mo --input_model <INPUT_MODEL>.nnet
```

## Using Kaldi-Specific Conversion Parameters <a name="kaldi_specific_conversion_params"></a>

The following list provides the Kaldi-specific parameters.

```sh
Kaldi-specific parameters:
  --counts COUNTS       A file name with full path to the counts file or empty string to utilize count values from the model file
  --remove_output_softmax
                        Removes the Softmax that is the output layer
  --remove_memory       Remove the Memory layer and add new inputs and outputs instead
```

## Examples of CLI Commands

* To launch Model Optimizer for the `wsj_dnn5b_smbr` model with the specified `.nnet` file:
   ```sh
   mo --input_model wsj_dnn5b_smbr.nnet
   ```

* To launch Model Optimizer for the `wsj_dnn5b_smbr` model with the existing file that contains counts for the last layer with biases:
   ```sh
   mo --input_model wsj_dnn5b_smbr.nnet --counts wsj_dnn5b_smbr.counts
   ```

  * The Model Optimizer normalizes сounts in the following way:
	\f[
	S = \frac{1}{\sum_{j = 0}^{|C|}C_{j}}
	\f]
	\f[
	C_{i}=log(S*C_{i})
	\f]
	where \f$C\f$ - the counts array, \f$C_{i} - i^{th}\f$ element of the counts array,
	\f$|C|\f$ - number of elements in the counts array;
  * The normalized counts are subtracted from biases of the last or next to last layer (if last layer is SoftMax).

     > **NOTE**: Model Optimizer will show a warning if a model contains values of counts and the `--counts` option is not used.

* If you want to remove the last SoftMax layer in the topology, launch the Model Optimizer with the
`--remove_output_softmax` flag:
   ```sh
   mo --input_model wsj_dnn5b_smbr.nnet --counts wsj_dnn5b_smbr.counts --remove_output_softmax
   ```

   The Model Optimizer finds the last layer of the topology and removes this layer only if it is a SoftMax layer.

   > **NOTE**: Model Optimizer can remove SoftMax layer only if the topology has one output.

* You can use the *OpenVINO Speech Recognition* sample application for the sample inference of Kaldi models. This sample supports models with only one output. If your model has several outputs, specify the desired one with the `--output` option.

## Converting a Model for Intel® Movidius™ Myriad™ VPU

If you want to convert a model for inference on Intel® Movidius™ Myriad™ VPU, use the `--remove_memory` option.
It removes the Memory layers from the OpenVINO IR files. Additional inputs and outputs will appear in the IR files instead.
Model Optimizer will output the mapping between inputs and outputs. For example:
```sh
[ WARNING ]  Add input/output mapped Parameter_0_for_Offset_fastlstm2.r_trunc__2Offset_fastlstm2.r_trunc__2_out -> Result_for_Offset_fastlstm2.r_trunc__2Offset_fastlstm2.r_trunc__2_out
[ WARNING ]  Add input/output mapped Parameter_1_for_Offset_fastlstm2.r_trunc__2Offset_fastlstm2.r_trunc__2_out -> Result_for_Offset_fastlstm2.r_trunc__2Offset_fastlstm2.r_trunc__2_out
[ WARNING ]  Add input/output mapped Parameter_0_for_iteration_Offset_fastlstm3.c_trunc__3390 -> Result_for_iteration_Offset_fastlstm3.c_trunc__3390
```
Based on this mapping, link inputs and outputs in your application manually as follows:

1. Initialize inputs from the mapping as zeros in the first frame of an utterance.
2. Copy output blobs from the mapping to the corresponding inputs. For example, data from `Result_for_Offset_fastlstm2.r_trunc__2Offset_fastlstm2.r_trunc__2_out`
must be copied to `Parameter_0_for_Offset_fastlstm2.r_trunc__2Offset_fastlstm2.r_trunc__2_out`.

## Supported Kaldi Layers
For the list of supported standard layers, refer to the [Supported Framework Layers ](@ref openvino_docs_MO_DG_prepare_model_Supported_Frameworks_Layers) page.

## Additional Resources
See the [Model Conversion Tutorials](@ref openvino_docs_MO_DG_prepare_model_convert_model_tutorials) page for a set of tutorials providing step-by-step instructions for converting specific Kaldi models. Here are some examples:
* [Convert Kaldi ASpIRE Chain Time Delay Neural Network (TDNN) Model](@ref openvino_docs_MO_DG_prepare_model_convert_model_kaldi_specific_Aspire_Tdnn_Model)

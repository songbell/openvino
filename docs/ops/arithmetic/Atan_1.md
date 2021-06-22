## Atan <a name="Atan"></a> {#openvino_docs_ops_arithmetic_Atan_1}

**Versioned name**: *Atan-1*

**Category**: Arithmetic unary operation

**Short description**: *Atan* performs element-wise inverse tangent (arctangent) operation with given tensor.

**Detailed description**:  Operation takes one input tensor and performs the element-wise inverse tangent function on a given input tensor, based on the following mathematical formula:

\f[
a_{i} = atan(a_{i})
\f]

**Attributes**:

    No attributes available.

**Inputs**

* **1**: An tensor of type T. **Required.**

**Outputs**

* **1**: The result of element-wise atan operation. A tensor of type T.

**Types**

* *T*: any numeric type.

**Examples**

*Example 1*

```xml
<layer ... type="Atan">
    <input>
        <port id="0">
            <dim>256</dim>
            <dim>56</dim>
        </port>
    </input>
    <output>
        <port id="1">
            <dim>256</dim>
            <dim>56</dim>
        </port>
    </output>
</layer>
```

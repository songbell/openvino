## Ceiling <a name="Ceiling"></a> {#openvino_docs_ops_arithmetic_Ceiling_1}

**Versioned name**: *Ceiling-1*

**Category**: Arithmetic unary operation

**Short description**: *Ceiling* performs element-wise ceiling operation with given tensor.

**Detailed description**: For each element from the input tensor calculates corresponding
element in the output tensor with the following formula:

\f[
a_{i} = ceiling(a_{i})
\f]

**Attributes**: *Ceiling* operation has no attributes.

**Inputs**

* **1**: A tensor of type *T* and arbitrary shape. **Required.**

**Outputs**

* **1**: The result of element-wise ceiling operation. A tensor of type *T*.

**Types**

* *T*: any numeric type.


**Examples**

*Example 1*

```xml
<layer ... type="Ceiling">
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

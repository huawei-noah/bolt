# Play around with Bolt in Python

## 1. Compile the Library

For example, do the compilation in Linux. After compiling bolt, you will find the shared library(.so) named "bolt.cpython-36m-x86_64-linux-gnu.so".  Switch to the path in which the shared library(.so) is stored , and check whether the shared library(.so) is available.

```
cd /bolt

./install.sh --target=linux-x86_64_avx2 -t 32 --python_api

cd /bolt/install_linux-x86_64_avx2/lib

python3
>>> import bolt
>>> bolt.__version___
'1.3.1'
>>>
```

## 2. Check the Python APIs

```
PYBIND11_MODULE(bolt, m)
{
    m.doc() = "bolt python class";

    m.attr("__version__") = "1.3.1";

    py::class_<Bolt>(m, "Bolt")
        .def(py::init<>())
        .def("convert", &Bolt::convert, py::arg("modelDirectory"), py::arg("modelName"),
            py::arg("precision"))
        .def("set_num_threads", &Bolt::set_num_threads, py::arg("threads"))
        .def("load", &Bolt::load, py::arg("boltModelPath"), py::arg("affinity"))
        .def("get_input_info", &Bolt::get_input_info)
        .def("get_output_info", &Bolt::get_output_info)
        .def("infer", &Bolt::infer);
}
```

All the python APIs are defined in "./inference/engine/src/bolt_python.cpp"  

## 3. Run the python example

```
>>> import bolt
>>> import numpy as np

>>> # 0. initialize bolt object && set threads num
>>> resnet18 = bolt.Bolt()
>>> resnet18.set_num_threads(4)

>>> # 1. convert the onnx model to bolt
>>> # Assuming that /path/resnet18_v2_sim.onnx exists.
>>> model_path = resnet18.convert(modelDirectory="/path", modelName="resnet18_v2_sim", precision="FP32")

>>> # 2. load model with affinity
>>> # affinity should be {'CPU_HIGH_PERFORMANCE','CPU_LOW_POWER','GPU','CPU'}
>>> resnet18.load(model_path, "CPU_HIGH_PERFORMANCE")

>>> # 3. check the model input information
>>> resnet18.get_input_info()
{'data': [1, 3, 224, 224]}

>>> # 4. infer with all 1 data
>>> all_ones_data = {"data": [1.0] * (1*3*224*224)}
>>> result1 = resnet18.infer(all_ones_data)

>>> # 5. infer with random data
>>> random_data = {"data": np.random.rand(1*3*224*224).astype(float).tolist()}
>>> result2 = resnet18.infer(random_data)


>>> # initialize another bolt object
>>> lenet = bolt.Bolt()
>>> # Assuming that /path/lenet_sim.onnx exists.
>>> model_path = lenet.convert(modelDirectory="/path", modeName="lenet_sim", precision="FP32")
>>> lenet.load(model_path, 'CPU_LOW_POWER')
>>> lenet.get_input_info()
{'import/Placeholder:0': [1, 1, 28, 28]}

>>> all_ones_data = {"import/Placeholder:0": [1.0] * (1*1*28*28)}
>>> result3 = lenet.infer(all_ones_data)
```

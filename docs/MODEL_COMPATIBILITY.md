# Model Compatibility

## ModelSpec change

not supported

## Operator parameter change

* change parameter size query function *get_operator_parameter_size* in [common/uni/include/parameter_spec.h](../common/uni/include/parameter_spec.h).

for example:
```
if (version < 20220817) {
    if (operatorType == OT_Resize) {
        size = size - sizeof(float) - 2 * sizeof(int);
    }
}
```

* change parameter's variable initialization function *model_compatibility* in [common/model_spec/src/model_deserialize.cpp](../common/model_spec/src/model_deserialize.cpp).

for example:
```
if (version < 20220817) {
    if (p->type == OT_Resize) {
        p->ps.resize_spec.zoom_factor = 0;
        p->ps.resize_spec.pad_begin = 0;
        p->ps.resize_spec.pad_end = 0;
    }
}
```

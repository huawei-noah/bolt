## Input Information

| input name | color | format | dimension | data order |
| ---------- | ----- | ------ | --------- | ---------- |
| image | RGB | NHWC | (1,3,720,1280) | 720x1280x3 |
| image | RGB | NCHW | (1,720,1280,3) | 720x1280x3 |


## Parameter List

| src format | dst format | dst data type |
| ---------- | ------ | --------- |
| YUV_NV21 | RGB_0_1 (data range:0~1)     | float |
| YUV_NV21 | RGB_0_255 (data range:0~255) | float, uint8 |
| YUV_NV21 | BGR_0_1                      | float |
| YUV_NV21 | BGR_0_255                    | float, uint8 |
| YUV_NV21 | RGBA_0_1                     | float |
| YUV_NV21 | RGBA_0_255                   | float, uint8 |
| YUV_NV21 | BGRA_0_1                     | float |
| YUV_NV21 | BGRA_0_255                   | float, uint8 |
| RGB_0_1    | YUV_NV21 | uint8 |
| RGB_0_255  | YUV_NV21 | uint8 |
| BGR_0_1    | YUV_NV21 | uint8 |
| BGR_0_255  | YUV_NV21 | uint8 |
| RGBA_0_1   | YUV_NV21 | uint8 |
| RGBA_0_255 | YUV_NV21 | uint8 |
| BGRA_0_1   | YUV_NV21 | uint8 |
| BGRA_0_255 | YUV_NV21 | uint8 |

[rgb_uint8.prototxt](./rgb_uint8.prototxt) show an example.

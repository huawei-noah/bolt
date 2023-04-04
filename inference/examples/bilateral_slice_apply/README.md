bilateral_slice_apply has 2 or 3 inputs, because guide is optional, image and grid are required.
we only give prototxt file in our example, and you can get caffemodel by renaming an exist model file.

## bilateral_slice_apply.prototxt show an example of 3 inputs.

| input name | format | dimension | data order |
| ---------- | ------ | --------- | ---------- |
| image | NHWC | (1,3,720,1280) | 720x1280x3 |
| grid | NHWC | (1,192,32,32) | 32x32x16x12 |
| guide | NHWC | (1,1,720,1280) | 720x1280x1 |


## bilateral_slice_apply_guide.prototxt show an example of 2 inputs. You can change guide calculation by modifying [compute/tensor/src/gpu/mali/cl/bilateral_slice_apply_c12.cl](../../../compute/tensor/src/gpu/mali/cl/bilateral_slice_apply_c12.cl).

| input name | format | dimension | data order |
| ---------- | ------ | --------- | ---------- |
| image | NHWC | (1,3,720,1280) | 720x1280x3 |
| grid | NHWC | (1,192,32,32) | 32x32x16x12 |

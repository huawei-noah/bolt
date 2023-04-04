#!/usr/bin/python
from torch.onnx import register_custom_op_symbolic

#def my_grid_sampler(g, self):
#    return g.op("com.microsoft::GridSample", self)
#register_custom_op_symbolic('::grid_sampler', my_grid_sampler, 11)

def grid_sample(g, input, grid, mode, padding_mode, align_corners):
    #return g.op("com.microsoft::GridSample", a, b, c, d, e)
    mode = sym_help._maybe_get_const(mode, "i")
    padding_mode = sym_help._maybe_get_const(padding_mode, "i")
    mode_str = ['bilinear', 'nearest', 'bicubic'][mode]
    padding_mode_str = ['zeros', 'border', 'reflection'][padding_mode]
    align_corners = int(sym_help._maybe_get_const(align_corners, "b"))
    return g.op("com.microsoft::GridSample", input, grid,
                    mode_s=mode_str,
                    padding_mode_s=padding_mode_str,
                    align_corners_i=align_corners)
register_custom_op_symbolic('::grid_sampler', grid_sample, 13)

if __name__ == '__main__':
    ckpt_path = "./checkpoint.pth"
    output_path = "./example.onnx"
    model = UNetPointRend(channel_depth=2, n_channels=3, n_classes=1)
    loaded_model = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(loaded_model, False)
    dummy_input = Variable(torch.randn(1, 3, 256, 256))
    torch.onnx.export(
        model, dummy_input, output_path,
        opset_version=13,
        verbose=True,
        do_constant_folding=True,
        custom_opsets={"com.microsoft": 1},
        input_names = ['input'], output_names = ['output']
    )

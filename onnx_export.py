""" ONNX export script

Export PyTorch models as ONNX graphs.

This export script originally started as an adaptation of code snippets found at
https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html

The default parameters work with PyTorch 1.6 and ONNX 1.7 and produce an optimal ONNX graph
for hosting in the ONNX runtime (see onnx_validate.py). To export an ONNX model compatible
with caffe2 (see caffe2_benchmark.py and caffe2_validate.py), the --keep-init and --aten-fallback
flags are currently required.

Older versions of PyTorch/ONNX (tested PyTorch 1.4, ONNX 1.5) do not need extra flags for
caffe2 compatibility, but they produce a model that isn't as fast running on ONNX runtime.

Most new release of PyTorch and ONNX cause some sort of breakage in the export / usage of ONNX models.
Please do your research and search ONNX and PyTorch issue tracker before asking me. Thanks.

Copyright 2020 Ross Wightman
"""
import argparse

import timm
from timm.utils.model import reparameterize_model
from timm.utils.onnx import onnx_export
import onnx
from onnxsim import simplify

parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('--output', metavar='ONNX_FILE',default='model.onnx',
                    help='output model filename')
parser.add_argument('--model', '-m', metavar='MODEL', default='regnety_008_tv',
                    help='model architecture (default: regnety_008_tv:RegNetY-800MF)')
# parser.add_argument('--model', '-m', metavar='MODEL', default='RegNetY-400MF',
#                     help='model architecture (default: RegNetY-400MF)')
parser.add_argument('--opset', type=int, default=None,
                    help='ONNX opset to use (default: 10)')
parser.add_argument('--keep-init', action='store_true', default=False,
                    help='Keep initializers as input. Needed for Caffe2 compatible export in newer PyTorch/ONNX.')
parser.add_argument('--aten-fallback', action='store_true', default=False,
                    help='Fallback to ATEN ops. Helps fix AdaptiveAvgPool issue with Caffe2 in newer PyTorch/ONNX.')
parser.add_argument('--dynamic-size', action='store_true', default=False,
                    help='Export model width dynamic width/height. Not recommended for "tf" models with SAME padding.')
parser.add_argument('--check-forward', action='store_true', default=False,
                    help='Do a full check of torch vs onnx forward after export.')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 1)')
# parser.add_argument('--img_size', default=None, type=int,
#                     metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--img_size', default=224, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float,  nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--num-classes', type=int, default=1000,
                    help='Number classes in dataset')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to checkpoint (default: none)')
parser.add_argument('--reparam', default=False, action='store_true',
                    help='Reparameterize model')
parser.add_argument('--training', default=False, action='store_true',
                    help='Export in training mode (default is eval)')
parser.add_argument('--verbose', default=False, action='store_true',
                    help='Extra stdout output')

def main():
    args = parser.parse_args()

    # args.pretrained = True
    args.pretrained = False
    if args.checkpoint:
        args.pretrained = False

    print("==> Creating PyTorch {} model".format(args.model))
    # NOTE exportable=True flag disables autofn/jit scripted activations and uses Conv2dSameExport layers
    # for models using SAME padding
    model = timm.create_model(
        args.model,   #模型名
        num_classes=args.num_classes,
        in_chans=3,
        pretrained=args.pretrained,
        checkpoint_path=args.checkpoint,
        exportable=True,
    )
    print(model)

    if args.reparam:
        model = reparameterize_model(model)

    onnx_export(
        model,
        args.output,
        opset=args.opset,
        dynamic_size=args.dynamic_size,
        aten_fallback=args.aten_fallback,
        keep_initializers=args.keep_init,
        check_forward=args.check_forward,
        training=args.training,
        verbose=args.verbose,
        input_size=(3, args.img_size, args.img_size),
        batch_size=args.batch_size,
    )

    # 加载导出的 ONNX 模型
    onnx_model = onnx.load(args.output)
    # 简化模型
    simplified_model, check = simplify(onnx_model)  #会去除一些无用节点和边，以减少模型的复杂性和计算负担,但也可能丢失一些细节信息。
    # 保存简化后的模型
    onnx.save_model(simplified_model, "simplified_model.onnx")    

if __name__ == '__main__':
    main()

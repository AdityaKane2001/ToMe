# python validate.py /workspace/datasets/ImageNet/val \
#     --model="tome_vit_base_patch16_224"\
#     --pretrained

python validate.py --data-dir /workspace/datasets/ImageNet/ \
    --model "tome_vit_base_patch16_224_augreg_in21k_ft_in1k" \
    --pretrained
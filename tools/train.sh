# python tools/train.py /home/ct/code/fgvc/mmclassification/configs/fgvc/greedyhash_vit-base-p16_pt-64xb64_iBioHash1k-224.py \
#     --work-dir './results/baseline' \
#     --seed 42 \
#     --no-validate \
#     --deterministic \

bash ./tools/dist_train.sh /home/ct/code/fgvc/mmclassification/configs/fgvc/greedyhash_vit-base-p16_pt-64xb64_iBioHash1k-224.py 2 \
    --work-dir './results/greedyhash' --amp --cfg-options seed=42+deterministic
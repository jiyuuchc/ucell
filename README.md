# UCell
A small cell segmentation model with not so small generalizability.

Ref: https://doi.org/10.48550/arXiv.2604.00243

## Try it
```
docker run --gpus=all -p 50051:50051 jiyuuchc/ucell:latest
```
This creates an GRPC endpoint at port 50051, which you can visit using [GUI clients](https://github.com/biopb), e.g. [napari-biopb](https://github.com/biopb/napari-biopb).

If you prefer programatic interface: see proto document [here](https://buf.build/jiyuuchc/biopb) and code example [here](https://github.com/biopb/biopb/tree/main/src/examples).


## Fine-tuning
We recommend this setting
```
python train.py \
  --init ${BASE_MODEL} \
  --config config.py:train \
  --config.data_dir=${DATADIR} \
  --config.n_iters=1 \
  --config.epochs_per_iter=1024 \
  --config.ema_decay=0.95 \
  --config.lora.rank=16
``` 
Save your training data (*.tif, *_label.tif) under ${DATADIR}/train.

Based model weights can be downloaded [here](https://huggingface.co/jiyuuchc/ucell/tree/main).

---

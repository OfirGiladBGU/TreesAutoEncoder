# Source of AE

https://github.com/dariocazzani/pytorch-AE

# Graph Generators

https://github.com/networkx/grave

https://github.com/deyuan/random-graph-generator

https://github.com/mlimbuu/random-graph-generator

https://github.com/mlimbuu/TCGRE-graph-generator

https://github.com/connectedcompany/alph


# VGG loss

https://github.com/crowsonkb/vgg_loss/tree/master

---

# Data info:

1. parse2022 `labels`, `preds` -> Values: binary (0, 255), Dim: (..., ..., ...)
2. parse2022 `preds_compnents` -> Values: grayscale (0-255), Dim: (..., ..., ...)
3. mini_cropped `labels`, `preds` -> Values: binary (0, 1), Dim: (..., ..., ...)


# The available approaches:

Given the `3d ground truth` and the `3d predicted labels`:

1. (Main Core Flow) Option 1 Flows: 
   1. Crop and Project both the `3d ground truth` and `3d predicted labels`:
      1. 6 views of `2d ground truth`
      2. 6 views of `2d predicted labels`
   2. Use `model1` as follows (2D to 2D):
      1. Train with the 6 `2d predicted labels` to **repair** and get 6 `2d ground truth`
      2. Predict with the 6 `2d predicted labels` to get the 6 `2d fixed labels`
   3. Use `model2` as follows (6-2D to 3D):
      1. (Seems better) Option 1:
         1. Train with the 6 `2d ground truth` to **reconstruct** and get the `3d ground truth`
         2. Predict with the 6 `2d fixed labels` to get the `3d fixed labels`
      2. (Might be problematic) Option 2:
         1. Train with the 6 `2d predicted labels`/`2d fixed labels`  to **reconstruct** and get the `3d ground truth`
         2. Predict with the 6 `2d fixed labels` to get the `3d fixed labels`
   4. Use all the `3d fixed label` to fix the `3d predicted labels`

2. (Secondary Core Flow) Option 2 Flows: 
   1. Crop and Project both the `3d ground truth` and `3d predicted labels` to `cropped`:
      1. 6 views of `2d ground truth`
      2. 6 views of `2d predicted labels`
   2. Use `model1` as follows (2D to 2D):
      1. Train with the 6 `2d predicted labels` to **repair** and get 6 `2d ground truth`
      2. Predict with the 6 `2d predicted labels` to get the 6 `2d fixed labels`
   3. Perform direct projection (using `logical or`) for all the data:
      1. From `2d ground truth` to get the `pre-3d ground truth`
      2. From `2d predicted labels` to get the `pre-3d predicted labels`
      3. From `2d fixed labels` to get the `pre-3d fixed labels`
   4. Use `model2` as follows (3D to 3D):
      1. (Seems better) Option 1:
         1. Train with the `pre-3d ground truth` to **reconstruct** and get the `3d ground truth`
         2. Predict with the `pre-3d fixed labels` to get the `3d fixed labels`
      2. (Might be problematic) Option 2:
         1. Train with the `pre-3d predicted labels`/`pre-3d fixed labels` to **reconstruct** and get the `3d ground truth`
         2. Predict with the `pre-3d fixed labels` to get the `3d fixed labels`
   5. Use all the `3d fixed label` to fix the `3d predicted labels`

3. (Direct Repair Flow) Option 3 Flows:
   1. Crop both the `3d ground truth` and `3d predicted labels`:
      1. `cropped 3d ground truth`
      2. `cropped 3d predicted labels`
   2. Use `model` as follows (3D to 3D):
      1. Train with the `cropped 3d predicted labels` to the `cropped 3d predicted labels` to **reconstruct** and get the `cropped 3d ground truth`
      2. Predict with the `cropped 3d predicted labels` to get the `cropped 3d fixed labels`
   3. Use all the `cropped 3d fixed label` to fix the `3d predicted labels`

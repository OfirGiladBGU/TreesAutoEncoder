# Trees Auto Encoder

Commit with big change:
11.04.2025 - configured new noise filters based on the Continuity fix 3D filters


## Config info:
```
# Parse2022:
None

# PipeForge3D - OBJ:
mesh_scale = 0.25
voxel_size = 1.0

# PipeForge3D - PCD:
points_scale = 0.25
voxel_size = 1.0

# Hospital CUP:
points_scale = ~50
voxel_size = 1.0
```


## How to create the dataset (Example for Parse2022):

1. Put the `parse2022` dataset with the `labels` and `preds` data on the path: `./data/parse2022`
   - `labels` - The ground truth
   - `preds` - The predicted labels by a model
2. Create config file in `.yaml` format and put it in `configs` folder (see example: [parse2022.yaml](configs/parse2022.yaml)).
3. Build the `dataset_2d` (also `dataset_1d`) by running `dataset_2d_creator` script: [dataset_2d_creator.py](datasets_forge/dataset_2d_creator.py)
4. Build the `dataset_3d` by running  `dataset_3d_creator` script: [dataset_3d_creator.py](datasets_forge/dataset_3d_creator.py)


## How To train the models:

- To train the `model_1d` run `main_1d` script: [main_1d.py](main_1d.py)
- To train the `model_2d` run `main_2d` script: [main_2d.py](main_2d.py)
- To train the `model_3d` run `main_3d` script: [main_3d.py](main_3d.py)

**Notice:** all scripts are calling the generic [main_base.py](main_base.py) scripts that call the matching training 
scripts in the files, based on the `model_type` parameter in the relevant main script:

- [train_1d.py](trainer/train_1d.py)
- [train_2d.py](trainer/train_2d.py)
- [train_3d.py](trainer/train_3d.py)

## How to test the full pipeline:

1. Run the `predict_pipeline` script: [predict_pipeline.py](predict_pipeline.py)


## Generate synthetic data with holes (Example for PipeForge3D):
1. Put the `PipeForge3D` dataset with the `originals` data on the path: `./data/PipeForge3D`
   - `originals` - The raw data, before voxelization
2. Run either of the following scripts depending on your data type:
   - [generate_3d_preds_from_mesh.py](datasets_forge/generate_3d_preds_from_mesh.py) - For mesh like data types: `_mesh.ply`, `.obj`, `.nii.gz` (annotations)
   - [generate_3d_preds_from_pcd.py](datasets_forge/generate_3d_preds_from_pcd.py) - For point clound like data types: `_pcd.ply`, `.pcd`
3. Repeat the steps in `How to create the dataset`.

---

# Data info:

1. parse2022 `labels`, `preds` -> Values: binary {0, 1}, Dim: 3
2. parse2022 `preds_compnents` -> Values: grayscale (0-255), Dim: 3
3. cropped 2d `labels`, `preds` -> Values: grayscale (0-255), Dim: 2
4. cropped 2d `components` -> Values: RGB (0-255, 0-255, 0-255), Dim: 2
5. cropped 3d `labels`, `preds` -> Values: binary {0, 1}, Dim: 3
6. cropped 3d `components` -> Values: grayscale (0-255), Dim: 3


# The Available Approaches:

Given the `3d ground truth` and the `3d predicted labels`:

1. (Main Core Flow) Option 1 Flows: 
   1. Crop and Project both the `3d ground truth` and `3d predicted labels`:
      1. 6 views of `2d ground truth`
      2. 6 views of `2d predicted labels`
   2. Use `model1` as follows (2D to 2D):
      1. Train with the 6 `2d predicted labels` to **repair** and get 6 `2d ground truth`
      2. Predict with the 6 `2d predicted labels` to get the 6 `2d fixed labels`
   3. Use `model2` as follows (6-2D to 3D):
      1. Train with the 6 `2d ground truth` to **reconstruct** and get the `3d ground truth`
      2. Predict with the 6 `2d fixed labels` to get the `3d fixed labels`
      
   4. Use all the `3d fixed label` to fix the `3d predicted labels`

2. (Secondary Core Flow) Option 2 Flows: 
   1. Crop and Project both the `3d ground truth` and `3d predicted labels` to `cropped`:
      1. 6 views of `2d ground truth`
      2. 6 views of `2d predicted labels`
   2. Use `model1` as follows (2D to 2D):
      1. Train with the 6 `2d predicted labels` to **repair** and get 6 `2d ground truth`
      2. Predict with the 6 `2d predicted labels` to get the 6 `2d fixed labels`
      * **2 approaches available with the same data:**
         1. Option 1: work with batches of 1 view (size: (b, 1, w, h))
         2. Option 2: work with batches of 6 views (size: (b, 6, 1, w, h))
   3. Perform direct projection (using `logical or`) for all the data:
      1. Option 1:
         1. From `2d ground truth` to get the `pre-3d ground truth`
         2. From `2d predicted labels` to get the `pre-3d predicted labels`
         3. From `2d fixed labels` to get the `pre-3d fixed labels`
      2. Option 2:
         1. From `2d ground truth` to get the `pre-3d ground truth`
         2. From `2d predicted labels` to get the `pre-3d predicted labels`
         3. From `2d fixed labels` to get the `pre-3d fixed labels`
         4. Merge `3d predicted labels` and `pre-3d ground truth` to get the `fused pre-3d ground truth`
         5. Merge `3d predicted labels` and `pre-3d fixed labels` to get the `fused pre-3d fixed labels`
   4. Use `model2` as follows (3D to 3D):
      1. Option 1 (Fill the whole internal space):
         1. Train with the `pre-3d ground truth` to **reconstruct** and get the `3d ground truth`
         2. Predict with the `pre-3d fixed labels` to get the `3d fixed labels`
      2. Option 2 (Fill only the predicted labels):
         1. Train with the `fused pre-3d ground truth` to **reconstruct** and get the `3d ground truth`
         2. Predict with the `fused pre-3d fixed labels` to get the `3d fixed labels`
   5. Use all the `3d fixed label` to fix the `3d predicted labels`

3. (Direct Repair Flow) Option 3 Flows:
   1. Crop both the `3d ground truth` and `3d predicted labels`:
      1. `cropped 3d ground truth`
      2. `cropped 3d predicted labels`
   2. Use `model` as follows (3D to 3D):
      1. Train with the `cropped 3d predicted labels` to the `cropped 3d predicted labels` to **reconstruct** and get the `cropped 3d ground truth`
      2. Predict with the `cropped 3d predicted labels` to get the `cropped 3d fixed labels`
   3. Use all the `cropped 3d fixed label` to fix the `3d predicted labels`

# Predict Pipeline Flow (Old):

1. Use model 1 on the `parse_preds_mini_cropped`
2. Save the results in `parse_fixed_mini_cropped`
3. Perform direct `logical or` on `parse_fixed_mini_cropped` to get `parse_prefixed_mini_cropped_3d`
4. Use model 2 on the `parse_prefixed_mini_cropped_3d`
5. Save the results in `parse_fixed_mini_cropped_3d`
6. Run steps 1-5 for mini cubes and combine all the results to get the final result
7. Perform cleanup on the final result (delete small connected components)

---

# Sources:


## Source of AE

https://github.com/dariocazzani/pytorch-AE


## Graph Generators

https://github.com/networkx/grave

https://github.com/deyuan/random-graph-generator

https://github.com/mlimbuu/random-graph-generator

https://github.com/mlimbuu/TCGRE-graph-generator

https://github.com/connectedcompany/alph


## VGG loss:

https://github.com/crowsonkb/vgg_loss/tree/master

## Tool to visualize cube:

https://3dthis.com/photocube.htm

## Configuring Matplotlib Plots to Display in a Window in PyCharm

See the following question on [Stack Overflow](https://stackoverflow.com/questions/57015206/how-to-show-matplotlib-plots-in-a-window-instead-of-sciview-toolbar-in-pycharm-p).

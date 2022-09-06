# Visualization / Evaluation

Before reading this doc, we hope that you've known how to run testing script.

## Visualization

For visualization, you can modify codes in [visualization](../depth_completion/visualize).

- [npy\_to\_mat](../depth_completion/visualization/npy_to_mat) : Codes in this directory simply transform `.npy` files to `.mat` files. Noted that we stronly recommend you using "original" raw depth, render depth rather than resized one for higher image quality and avoiding interpolation.
- [visualization.m](../depth_completion/visualization/visualization.m) : Codes used to visualize depth map in jet color map. You can modify first 30 lines for different input and output data path.

## Evaluation

For evaluation, you only need to execute the following scripts :

```bash
python3 -m depth_completion.eval <output_dir>
```


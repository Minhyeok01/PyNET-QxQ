PyNET-QxQ takes 16-bit or 8-bit `PNG` images as inputs.
A third-party image viewer is required to handle 16-bit `PNG` images.
 
# [Usage guide]
# Step.1: Initial Setting
     # Requirements
     See `requirements.txt`.
     
     # Data
     `PNG` images should be located in `./raw_images/test/`.
     The `test` folder consists of `msc` (input images) and `gt` (target output images).
     To compute validation metrics, you must exclude QxQ images from test (since QxQ images do not have ground truth images).

     # Parameters
     You should put the parameter file in the `./Experiments/models` folder with the name `best_acc_model.pth`.

# Step.2: Model Test
    1) Run in cpu-only mode:
     ```
     python main.py --cpu
     ```

    2) Run with cuda:
     ```
     python main.py
     ```

# Notice
    You can update options in `option.py` to modify settings.
    For example, to output validation metrics, you can activate `--metric` option. 
     ```
     python main.py --metric
     ```
    
    Test outputs are saved in './Experiments/test_images'
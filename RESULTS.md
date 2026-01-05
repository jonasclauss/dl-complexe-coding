## Participants
- Lukas Marche - 
- Robin Berge - 3792567
- Jonas Clauß - 

## Config Parameters:
In our project we have three different levels to set the parameters.

1. The default config:
    If none of the other levels is set the project runs with these default settings. These parameters can be found in the `helpers/config.py` file. Please don't change these settings.
2. The `config.json`file:
    In this config file you can choose your own parameters which override the default configs. The structure needs to look like this. All possible values are listed below:
    
    ```jsonc
    {
        // Seed for all RNGs (e.g. matriculation number).
        "seed": 3792567,

        // Base project path; data paths are resolved relative to this.
        "project_path": ".", 

        // Path to dataset root (e.g. ./data/EuroSAT_RGB or ./data/EuroSAT_MS)
        "data_path": "./data/EuroSAT_RGB",
    
        // Image source to use (*rgb* or *ms*). 
        "data_source": {"rgb", "ms"}, 
        
        // Number of training epochs.
        "epochs": 15,

        // Batch size for training and evaluation.
        "batch_size": 128,

        // Number of DataLoader workers.
        "workers": 4,

        // Learning rate.
        "learning_rate": 1e-4,
        
        // Weight decay for optimizer.
        "weight_decay": 0.01,

        // List of augmentations/preprocessing tags: {none, mild, strong, resnet}]. Combinations
        // are possible.
        "augmentation": [{"none", "mild", "strong", "resnet", }],

        // Model type (cnn or resnet).
        "model": {"cnn", "resnet"}, 

        // Run in reproduction mode (skip training, load model, check logits).
        "reproduction": {false, true},

        // Save computed logits (only in reproduction mode).
        "save_logits": {fals, true},

        // Path to the logits file.
        "logits_path": "logits.pt",

        // Path to save/load the model.
        "model_path": "model.pth"
    }
    ```

    If a parameter is not set, the default value is used.
3. Inline arguments:
    This is the highest level of configuration and overrides the other levels. You can see all possble inline arguments with `--help` argument while executing the `main.py` eg. `python ./main.py --help`.

## Compute the prediction and compare the logits
To run the reproduction routine and compare the predictions you need to run the following command:
```shell
python main.py --reproduction --model-path <path-to-model> --logits-path <path-to-baseline-logit>
```
It runs the test test on the given model in the `--model-path` (default is the `model.pth`). The results are printed out in the console. Afterwards it compares it with a given logit which can be provided with the `--logits-path` argument (default is `logits.pt`).
If you want to save the logits new, you can add the argument `--save-logits`.

## Results
In the following we will present the results of the validation sets during training.

![MildSelf]()
![StrongSelf]()
![MildResNet]()
![StrongResNet]()
![PreProcessResNet]()

## Top and Bottom Images
In the following you can see the top-5 and bottom-5 images for the classes AnnualCrop, Forest and Hebaceous Vegetation:\newline
![AnnualCrop](./images/ranking_class_0_AnnualCrop.png)
![Forest](./images/ranking_class_1_Forest.png)
![AnnualCrop](./images/ranking_class_2_HerbaceousVegetation.png)

This is just a simple implementation of the inference for the project Depth-Anything to perform metric depth estimation. Since authods do not offer functionality for evaluating their model on different datasets, some code needed to be added, in order to perform meric depth estimation of a test image. Follow the steps:

1. Clone the original Depth-Anything project 
   (https://github.com/LiheYoung/Depth-Anything/tree/main)

2. Clone this project. You will need some files from here to replace those of Depth Anything. Replace file `metric_depth/evaluate.py` and folder `metric_depth/zoedepth` of the original project with those from this one.

3. Add a new folder `checkpoints` with the following two models that you 'll need to download:
  - https://huggingface.co/spaces/LiheYoung/Depth-Anything/tree/main/checkpoints_metric_depth (get outdoor model)
  - https://huggingface.co/spaces/LiheYoung/Depth-Anything/blob/main/checkpoints/depth_anything_vitl14.pth
  The checkpoints folder should be inside the metric_depth folder.

4. Run 
`conda env create -f environment.yml`
then 
`conda activate depth_env`. 

  If there is any other dependency that has not been installed (and returns an error when running the script), download it with conda or pip.

5. Take all images you want to estimate and add them to the directory `Data/Run` preferably with numbers as indices.

6. Go to the root of the project and run the following command: 
  `python3 estimate_parameters.py`
  You will find the results in `parameters_results.txt`.
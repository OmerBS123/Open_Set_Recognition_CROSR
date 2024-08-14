Overview
This framework is implemented based on the article "Classification-Reconstruction Learning for Open-Set Recognition," which is attached to this repository. The CROSR framework is designed to train and evaluate a network for open-set recognition tasks.
Setup and Usage
Training
To train the network, set the is_training variable to True in the run_notebook file. This will initiate the training process and subsequent evaluation of the model.
Evaluation
If is_training is set to False, the notebook will load the pre-trained network from the model_weights folder and perform evaluation only. This allows for quick testing and validation of model performance without the need for retraining.
Hardware Requirements
CUDA: If CUDA is available, the notebook will complete execution in under 10 minutes.
Without CUDA: Expect approximately 17 minutes for the notebook to run.
Directory Structure
Ensure that the entire CROSR framework is located in the same directory as the run_notebook. This is the default configuration; however, check this if you have made any modifications to the file structure.
Output
All plots generated during the training process are saved in the Plots folder within the CROSR directory. These visualizations are critical for assessing the performance and behavior of the network over time.
Additional Information
For more details, refer to the article provided in the repository, which lays the theoretical foundation for the methods used in this framework.

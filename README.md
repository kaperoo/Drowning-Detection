# Neural Network-Based Drowning Detection System
University of Leeds Final Year Project - Kacper Roemer
This is the code for the drowning detection system proposed in the report.

To install the required libraries do:
`pip install -r requirements.txt`

The code for the integrated drowning detection system can be found in the 'code' directory. There are 4 versions of the final system, each employing different model:
`integrated_baseline.py`,
`integrated_cnnsingle.py`,
`integrated_cnnseq.py` and
`integrated_cnnrnn.py`.

Other than those scripts, there are the following directories:
* `models`: Saved models and their training scripts.
* `testing`: Test scripts used to evaluate models and results.
* `hp_tuning`: Hyperparameter tuning scripts and results.
* `datasets`: Scripts used to create PyTorch datasets and keypoints.
* `system_tests`: Integrated systems but with functions to measure their latency and jitter.
* `utils`: Scripts used for data handling and keypoint extraction with YOLOv7.

What is not included in this repository:
* YOLOv7 Scripts and models.
* The "Water Behavior" dataset.

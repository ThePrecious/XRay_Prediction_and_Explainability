# X-Ray Diagnosis AI Assistant
Repository containing FSDL project

## Proposal

The TorchXRayVision Project [1,2] provides several different models trained on a variety of  X-ray datasets, such as NIH's ChestX-ray8. In this project, we aim to build an interface using the tools described in the lectures/labs to allow users / medical practitioners to explore, visualize, and interpret model predictions. For example, methods such as "gifsplanation" build animations to investigate counterfactual explanations for model outputs.

References:
[1] https://huggingface.co/torchxrayvision
[2] https://arxiv.org/abs/2111.00595
[3] https://mlmed.org/gifsplanation/

How does your project go beyond the usual "train a model to do well on a static benchmark" typical in ML class projects?

As described above, our focus is on using well-developed pre-trained models to investigate model predictions. Our proposed project highlights the deployment aspect of the class and we are also considering concepts related to data annotation / data flywheel feedback loop. In addition, we are  interested in the possibility of using the baseline pre-trained models for transfer learning.


Installation:
Run setup.py and then demo.py.

This requires a working version of  ffmpeg that can be used via command-line.

Demo recording link:
https://youtu.be/UyuGoUuO9C8?list=PL1T8fO7ArWle-HwX6SkoQ3j_ol19P7tGT

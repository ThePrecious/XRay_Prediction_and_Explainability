import os
import torch
import torchxrayvision as xrv
import torchvision
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import pandas as pd

from captum.attr import IntegratedGradients, Saliency, InputXGradient
from .gifsplanation import attribution
from .en_model import ChestXrayEnsemble

n_classes = 14
base_learner_names = [  'densenet121-res224-all',
                        'resnet50-res512-all',
                        'densenet121-res224-pc',
                        'densenet121-res224-rsna',
                        'densenet121-res224-chex',
                        'densenet121-res224-mimic_ch',
                        'densenet121-res224-mimic_nb',
                        'densenet121-res224-nih'
                     ]

chk_path = './src/model/epoch=00.ckpt'
checkpoint = torch.load(chk_path, map_location='cpu')

model_ensemble = ChestXrayEnsemble(num_classes=n_classes,base_learners=base_learner_names)
model_ensemble.load_state_dict(checkpoint['state_dict'])
model_ensemble.pathologies = model_ensemble.base_learners[7].pathologies
model_ensemble.eval()

def make_fig(plot_matrix):
    fig = plt.figure()
    plt.imshow(plot_matrix, cmap=plt.cm.hot)
    #plt.title(plot_title)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    return fig


def xrv_prepare_image(image):
    img = xrv.datasets.normalize(image, 255)  # convert 8-bit image to [-1024, 1024] range
    if img.ndim > 2:
      img = img.mean(2)[None, ...]  # Make single color channel
    else:
      img = img[None, ...]
    transform = torchvision.transforms.Compose(
        [
            xrv.datasets.XRayCenterCrop(),
            xrv.datasets.XRayResizer(224)
        ]
    )
    img = transform(img)
    img = torch.from_numpy(img)

    return img[None, ...]


### Prediction

def predict_ensemble(image):
    #pathologies = model_ensemble.base_learners[7].pathologies
    thresholds = [0.17, 0.87, 0.23, 0.96, 0.52, 0.99, 0.93, 0.28, 0.77, 0.13, 0.98, 0.16, 0.13, 0.08]
    scores = model_ensemble(image).detach().to('cpu').numpy()[0]
    scores = np.round(scores,4)
    diagnosis = ['Yes' if scores[i] > thresholds[i] else 'No' for i in range(n_classes)]
    result = [a for a in zip(model_ensemble.pathologies, scores, diagnosis) if a[0] != ""]
    d = {'Pathology': [a[0] for a in result], 'Score': [a[1] for a in result], 'Diagnosis': [a[2] for a in result]}
    df = pd.DataFrame(data=d).round(4)
    return df

def predict(image, model_choice):
    """Function that serves predictions."""
    img = xrv_prepare_image(image)
    if model_choice == 'torchxrayvision-ensemble':
        df = predict_ensemble(img)
    else:
        model = xrv.models.DenseNet(weights=model_choice)
        model.eval()
        outputs = model(img)
        scores =  outputs[0].detach().numpy().astype(np.float) #conversion to np.float is needed for visualization with gr.Label
        diagnosis = ['Yes' if scores[i] > 0.5 else 'No' for i in range(len(model.pathologies))]
        result = [a for a in zip(model.pathologies, scores, diagnosis) if a[0] != ""] #remove empty pathologies
        d = {'Pathology': [a[0] for a in result], 'Score': [a[1] for a in result], 'Diagnosis': [a[2] for a in result]}
        df = pd.DataFrame(data=d).round(4)
    return df


### Explanation

def explain_gradient(image, model_choice, target):
    """Function that serves explanations using the standard gradient-based saliency map"""
    if model_choice == 'torchxrayvision-ensemble':
      model = model_ensemble
    else:
      model = xrv.models.DenseNet(weights=model_choice)
    input = xrv_prepare_image(image)
    
    #Saliency
    saliency = Saliency(model)
    attr = saliency.attribute(input, target=model.pathologies.index(target))
    fig1 = make_fig( np.abs(attr[0,0].numpy()) )

    return fig1

def explain_input_x_gradient(image, model_choice, target):
    """Function that serves explanations using the input-times-gradient method."""
    
    if model_choice == 'torchxrayvision-ensemble':
      model = model_ensemble
    else:
      model = xrv.models.DenseNet(weights=model_choice)
    input = xrv_prepare_image(image)
    
    #InputXGradient
    ixg = InputXGradient(model)
    attr = ixg.attribute(input, target=model.pathologies.index(target))
    fig2 = make_fig( np.abs(attr[0,0].detach().numpy()) ) 
    return fig2
    
def explain_integrated_gradients(image, model_choice, target):
    """Function that serves explanations using integrated gradients"""
    
    if model_choice == 'torchxrayvision-ensemble':
      model = model_ensemble
    else:
      model = xrv.models.DenseNet(weights=model_choice)
    input = xrv_prepare_image(image)
    
    #IntegratedGradients
    ig = IntegratedGradients(model)
    attr = ig.attribute(input, target=model.pathologies.index(target))
    fig3 = make_fig( np.abs(attr[0,0].detach().numpy()) )
    return fig3

def explain_gifsplanation(image, model_choice, target):
    """Function that serves explanations using gifsplanation"""
    
    if model_choice == 'torchxrayvision-ensemble':
      model = model_ensemble
    else:
      model = xrv.models.DenseNet(weights=model_choice)
    input = xrv_prepare_image(image)
    
    #Gifsplanation
    input.requires_grad=False
    ae = xrv.autoencoders.ResNetAE(weights="101-elastic")

    pth = os.getcwd()

    movie = attribution.generate_video(input, model, target, ae,temp_path=pth+"/tmp", target_filename="test", border=False, show=False,
                        ffmpeg_path="ffmpeg")

    return movie

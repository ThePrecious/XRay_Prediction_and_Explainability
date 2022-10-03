
import torchxrayvision as xrv
import torch
import torchvision

import matplotlib.pyplot as plt
import numpy as np


def make_fig(plot_matrix):
    fig = plt.figure(figsize=(20, 20))
    plt.imshow(plot_matrix, cmap=plt.cm.hot)
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    return fig


def predict(image, model_choice=None, model=None):
    """Function that serves predictions."""
    img = xrv_prepare_image(image)
    if not model:
        model = xrv.models.DenseNet(weights=model_choice)
    model.eval()

    outputs = model(img)

    # conversion to np.float is needed for visualization with gr.Label
    scores = outputs[0].detach().numpy().astype(np.float)  
    label = dict(zip(model.pathologies, scores))
    return label


def xrv_prepare_image(image):
    img = xrv.datasets.normalize(image, 255)  # convert 8-bit image to [-1024, 1024] range
    img = img.mean(2)[None, ...]  # Make single color channel
    transform = torchvision.transforms.Compose(
        [
            xrv.datasets.XRayCenterCrop(),
            xrv.datasets.XRayResizer(224)
        ]
    )
    img = transform(img)
    img = torch.from_numpy(img)

    return img[None, ...]

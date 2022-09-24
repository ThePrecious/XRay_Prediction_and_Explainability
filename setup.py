from setuptools import setup

setup(
    name='xray-tools',
    author='',
    version='0.1',
    install_requires=[
        'scikit-image',
        'torch',
        'torchvision',
        'matplotlib',
        'torchxrayvision',
        'captum',
        'gradio'
    ],
)

from setuptools import setup, find_packages

setup(
    name='xray-tools',
    author='',
    version='0.1',
    packages=find_packages(),
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

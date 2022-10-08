from setuptools import setup, find_packages

setup(
    name='xray-tools',
    author='',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch>=1.12.1',
        'numpy>=1.21.6',
        'torchvision>=0.13.1',
        'matplotlib>=3.2.2',
        'torchxrayvision>=0.0.38',
        'captum>=0.5.0',
        'gradio>=3.4.1',
        'sklearn'
    ],
)

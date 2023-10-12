from setuptools import setup, find_packages

setup(
    name='samrunner',
    version='0.1.0',
    description='SAM wrapper for inferencing via MITK',
    author='Ashis Ravindran',
    author_email='ashis.ravindran@dkfz-heidelberg.de',
    url='mitk.org',
    packages=find_packages(include=['samrunner', 'samrunner.*']),
    install_requires=[
        'segment-anything @ git+https://github.com/facebookresearch/segment-anything.git@6fdee8f2727f4506cfbbe553e23b895e27956588',
        'SimpleITK>=2.2.1',
        'opencv-python>=4.7.0.72',
        'tqdm>=4.65.0'
    ],
    classifiers=[
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering',
            'Operating System :: Unix',
            'Operating System :: Microsoft :: Windows :: Windows 10'
            'Programming Language :: Python :: 3.8'
            'Programming Language :: Python :: 3.9'
            'Programming Language :: Python :: 3.10'
            'Development Status :: 4 - Beta'
            'Environment :: GPU :: NVIDIA CUDA'
        ],
    scripts=['samrunner/run_inference_daemon.py']
)

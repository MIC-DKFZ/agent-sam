from setuptools import setup, find_packages

setup(
    name='samrunner',
    version='0.1.0',
    description='SAM wrapper for inferencing via MITK',
    author='Ashis Ravindran',
    author_email='ashis.ravindran@dkfz.de',
    url='mitk.org',
    packages=find_packages(include=['samrunner', 'samrunner.*']),
    install_requires=[
        'medsam @ git+https://github.com/bowang-lab/MedSAM.git@2b7c64cf80bf1aba546627db9b13db045dd1cbab',
        'numpy<2',
        'SimpleITK>=2.2.1',
        'requests==2.27.1;python_version<"3.10"',
        'requests;python_version>="3.10"',
        'opencv-python>=4.7.0.72',
        'tqdm>=4.65.0'
    ],
    classifiers=[
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering',
            'Operating System :: Unix',
            'Operating System :: Microsoft :: Windows :: Windows 10'
            'Programming Language :: Python :: 3.9'
            'Programming Language :: Python :: 3.10'
            'Programming Language :: Python :: 3.11'
            'Programming Language :: Python :: 3.12'
            'Development Status :: 4 - Beta'
            'Environment :: GPU :: NVIDIA CUDA'
        ],
    scripts=['samrunner/run_inference_daemon.py', 'samrunner/sam_runner.py', 'samrunner/base_runner.py', 'samrunner/medsam_runner.py']
)

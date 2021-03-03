'''
Machine learning project for filename based media classification and named
entity recognition.
'''

import setuptools

with open('README.md', 'r') as f:
    LONG_DESCRIPTION = f.read()

setuptools.setup(
    name='mccore',
    version='1.2.3',
    author='Allan Wright',
    description='media-classifier-core package',
    long_description=LONG_DESCRIPTION,
    package_dir={'': 'src'},
    packages=setuptools.find_packages('src'),
    install_requires=[
        'numpy',
        'sklearn',
        'importlib_resources ; python_version<"3.7"',
        'spacy==2.3.5',
        'titlecase'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    include_package_data=True
)

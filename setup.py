# Copyright 2019 The SQLNet Company GmbH

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""setup.py for getml"""

from setuptools import setup, find_packages

long_description = """
    getML (https://get.ml) is a software for automated machine learning
    (AutoML) with a special focus on feature engineering for relational data
    and time series. The getML algorithms can produce features that are far
    more advanced than what any data scientist could write by hand or what you
    could accomplish using simple brute force approaches.

    This is the official python client for the getML engine.

    Documentation and more details at https://docs.get.ml/0.8
"""


setup(
    name="getml",
    version="0.8",
    author="getML",
    author_email="support@get.ml",
    url='https://docs.get.ml/0.8',
    download_url='https://github.com/getml/getml-python-api',
    description="Python API for getML",
    long_description=long_description,
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
    ],
    keywords = ['AutoML', 'feature engineering'],
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Telecommunications Industry',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Manufacturing',
        'Intended Audience :: Healthcare Industry',
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        'Development Status :: 4 - Beta',
    ],
)

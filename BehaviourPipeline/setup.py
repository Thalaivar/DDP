from setuptools import setup

setup(
    name='behaviourPipeline',
    version='0.1.0',
    author='Dhruv Laad',
    author_email='dhruvlaad@gmail.com',
    packages=['behaviourPipeline'],
    license='LICENSE.txt',
    install_requires=[
       "ray",
       "umap-learn",
       "hdbscan",
       "catboost",
       "pyyaml",
       "joblib",
       "scikit-learn"
    ],
)
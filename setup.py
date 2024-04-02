from setuptools import setup, find_packages

setup(
    name='cyclpept_ml_models',  # Name of your package
    version='0.1.0',  # Version number
    author='Your Name',
    author_email='your.email@example.com',
    description='Machine Learning models for cyclic peptides',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/AnnaKelmanson/cyclpept-ML-models',
    packages=find_packages(),  # Automatically find packages in your project
    install_requires=[
        # List your project dependencies here
        # e.g., 'numpy', 'pandas', 'scikit-learn'
    ],
    classifiers=[
        # Classifiers help users find your project
        # For a full list, see https://pypi.org/classifiers/
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Minimum version requirement of the package
)

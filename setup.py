from setuptools import setup, find_packages
import os

# Read requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Get long description from README if it exists
try:
    with open('README.md', 'r', encoding='utf-8') as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = 'A high-performance sign indexing library for similarity search'

setup(
    name='sign_indexing',
    version='0.1.0',
    description='A high-performance sign indexing library for similarity search',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Derrick Quinn',
    author_email='dq55@cornell.edu',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=requirements,
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='similarity-search indexing machine-learning',
    include_package_data=True,
    zip_safe=False,
)

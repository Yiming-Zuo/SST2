from setuptools import setup, find_packages

version = "0.0.1"

with open('README.rst', encoding='utf-8') as readme_file:
    readme = readme_file.read()

requirements = [
    'openmm>=7.7.0',
    'pdb_numpy>=0.0.1',
    'pandas>=1.3.0',
]

setup(
    name='SST2',
    version=version,
    description=(
        'SST2 is a python library designed to conduct Simulated',
        'Solute Tempering (SST2) simulations using the openmm library.'
    ),
    long_description=readme,
    long_description_content_type='text/markdown',
    author='Samuel Murail',
    author_email="samuel.murail@u-paris.fr",
    url='https://github.com/samuelmurail/SST2',
    packages=find_packages(),
    package_dir={'SST2': 'src/SST2'},
    entry_points={'console_scripts': ['SST2 = SST2.__main__:main']},
    include_package_data=True,
    python_requires='>=3.7',
    install_requires=requirements,
    license='GNUv2.0',
    zip_safe=False,
    classifiers=[
        "Development Status :: 5 - Devloppment/Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Programming Language :: Python",
        "Topic :: Software Development",
    ],
    keywords=[
        "SST2",
        "Python",
        "openmm",
    ],
)

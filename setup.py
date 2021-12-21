from setuptools import setup
import setuptools

setup(
    name='pykaruiflow',
    version='0.0.1',
    packages=setuptools.find_packages(),
    include_package_data=True,  # This will include all files in MANIFEST.in in the package when installing.
    classifiers=[
        'Programming Language :: Python :: 3.7',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ], install_requires=[]
)
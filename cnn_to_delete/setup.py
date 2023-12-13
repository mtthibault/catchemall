from setuptools import find_packages
from setuptools import setup

with open("cnn_requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(
    name="catchemall",
    version="0.0.1",
    description="Cath'em All",
    license="MIT",
    author="Batch 1437",
    # author_email="contact@lewagon.org",
    # url="https://github.com/mtthibault/catchemall",
    install_requires=requirements,
    packages=find_packages(),
    test_suite="tests",
    # include_package_data: to install data from MANIFEST.in
    include_package_data=True,
    zip_safe=False,
)

import subprocess

from setuptools import find_packages, setup

result = subprocess.Popen("pip install -r requirements.txt", shell=True)
result.wait()

setup(
    name="GenAIEval",
    version="0.0.0",
    author="Intel AISE AIPC Team",
    author_email="haihao.shen@intel.com, feng.tian@intel.com, chang1.wang@intel.com, kaokao.lv@intel.com",
    description="Evaluation and benchmark for Generative AI",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/opea-project/GenAIEval",
    packages=find_packages(),
    python_requires=">=3.10",
)

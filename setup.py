from setuptools import find_packages, setup


setup(
    name="moondream",
    version="0.2.0.dev",
    description="a tiny vision language model that kicks ass and runs anywhere",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="deep learning",
    license="Apache",
    author="vikhyatk",
    author_email="contact@moondream.ai",
    url="https://github.com/3DAlgoLab/moondream",
    packages=find_packages(include=["moondream", "moondream.*"]),
    entry_points={},
    python_requires=">=3.8.0",
    install_requires=[
        "accelerate",
        "huggingface-hub",
        "Pillow",
        "torch",
        "torchvision",
        "transformers",
        "einops",
        "gradio",
        "datasets",
    ],
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

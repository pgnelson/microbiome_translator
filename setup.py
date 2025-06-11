from setuptools import setup, find_packages

setup(
    name="microbiome_translator",
    version="0.1.0",
    description="Microbiome to Metabolite translator using attention-based PyTorch models",
    author="Your Name",
    packages=find_packages(where="src"),
	package_dir={"": "src"},
    install_requires=[
        "torch",
        "pandas",
        "numpy",
        "scipy",
        "statsmodels",
	"scikit-learn"
    ],
)

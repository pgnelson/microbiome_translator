from setuptools import setup, find_packages

setup(
	name="microbiome_translator",
	version="0.1.1",
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
	include_package_data=True,
	entry_points={
		"console_scripts": ["load_microbiome_model = examples.example_load_model:example_load",],
	}
)

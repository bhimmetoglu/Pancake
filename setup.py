from setuptools import setup

setup(name='PanCake',
	version='0.1',
	description='Model stacking package',
	long_description = open('README.md').read(),
	author='Burak Himmetoglu',
	packages = ['pancake'],
	license = 'MIT',
	install_requires = ['numpy', 'pandas', 'scikit-learn', 'joblib'],
	url = 'https://github.com/bhimmetoglu/Pancake',
	classifier = [ 'Programming Language :: Python :: 3']
)


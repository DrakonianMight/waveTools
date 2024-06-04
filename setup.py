import setuptools
setuptools.setup(name='waveTools',
version='1.2.1',
description='A package for reading and processing wave observations',
url='https://github.com/DrakonianMight/waveTools',
author='Leo Peach',
install_requires=['numpy', 'pandas','scipy','matplotlib', 'xarray', 'wavespectra'],
author_email='',
packages=setuptools.find_packages(where="src"),
package_dir={"":"src"},
python_requires=">=3.8",
zip_safe=False)
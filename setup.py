import time

from setuptools import find_packages, setup

install_requires = ['numpy', 'scipy','pandas','matplotlib']

setup(name='funfluid',
      version=time.strftime("%Y%m%d%H%M", time.localtime()),
      description='funfluid',
      author='bingtao',
      author_email='1007530194@qq.com',
      url='https://github.com/1007530194',

      packages=find_packages(),
      install_requires=install_requires
      )

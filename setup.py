import os
import re
from setuptools import find_packages, setup

install_requires = [
    'torch', #>=1.5.0
]

def _read(long_description_name):
    # read the contents of your README file
    this_directory = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(this_directory, long_description_name), encoding='utf-8') as f:
        return f.read()

def _read_version():
    regexp = re.compile(r"^__version__\W*=\W*'([\d.abrc]+)'")
    init_py = os.path.join(
        os.path.dirname(__file__), 'tmomentum', '__init__.py'
    )
    with open(init_py) as f:
        for line in f:
            match = regexp.match(line)
            if match is not None:
                return match.group(1)
        raise RuntimeError(
            'Cannot find version in tmomentum/__init__.py'
        )

setup(name='tmomentum',
      version=_read_version(),
      description=('t-momentum'),
      long_description=_read('README.md'),
      long_description_content_type='text/markdown',
      author='Wendyam Eric Lionel Ilboudo (Mahoumaru)',
      author_email='lionelwendyam@yahoo.fr',
      url='https://github.com/Mahoumaru/t-momentum.git',
      download_url='https://pypi.org/manage/project/tmomentum',
      install_requires=install_requires,
      packages=find_packages(),
)

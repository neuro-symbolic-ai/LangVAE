import os
from distutils.core import setup

PKG_ROOT = os.path.abspath(os.path.dirname(__file__))


def load_requirements() -> list:
    """Load requirements from file, parse them as a Python list"""

    with open(os.path.join(PKG_ROOT, "requirements.txt"), encoding="utf-8") as f:
        all_reqs = f.read().split("\n")
    install_requires = [str(x).strip() for x in all_reqs]

    return install_requires


setup(
    name='LangVAE',
    version='0.2.1',
    packages=['langvae', 'langvae.arch', 'langvae.decoders', 'langvae.encoders', 'langvae.data_conversion'],
    url='',
    license='',
    author='Danilo S. Carvalho',
    author_email='danilo.carvalho@manchester.ac.uk',
    description='',
    install_requires=load_requirements()
)

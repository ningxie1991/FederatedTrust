from setuptools import setup

with open('requirements.txt', 'r') as f:
    install_requires = f.read().splitlines()

setup(
    name='FederatedTrust',
    version='0.1.0',
    packages=['federatedTrust'],
    package_data={
        # If the federatedTrust package contains *.json, include them:
        'federatedTrust': ['*.json'],
    },
    license='MIT',
    author='ningx',
    install_requires=install_requires,
    author_email='ning.xie@uzh.ch',
)

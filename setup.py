from setuptools import setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='aws',
    url='https://github.com/awsaf49/my-package',
    author='awsaf rahman',
    author_email='awsaf49@gmail.com',
    # Needed to actually package something
    packages=['aws'],
    # Needed for dependencies
    install_requires=['numpy','panda'],
    # *strongly* suggested for sharing
    version='1.0',
    # The license can be anything you like
    license='MIT',
    description='An example of a python package from pre-existing code',
    # We will also need a readme eventually (there will be a warning)
    long_description=open('README.txt').read(),
)
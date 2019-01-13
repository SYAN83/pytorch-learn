from setuptools import setup
import re


APP_NAME = 'pytorch-learn'
VERSION = '0.1'


def readme():
    with open('README.rst', 'r') as f:
        return f.read()


def check_version():
    global VERSION
    # Get the version
    version_regex = r'__version__ = ["\']([^"\']*)["\']'
    with open('ptlearn/__init__.py', 'r') as f:
        text = f.read()
        match = re.search(version_regex, text)
        if match:
            VERSION = match.group(1)
        else:
            raise RuntimeError("No version number found!")


if __name__ == '__main__':
    check_version()

    setup(
        name=APP_NAME,
        version=VERSION,
        description='A Python machine learing library, based on PyTorch',
        long_description=readme(),
        classifiers=[
            'Development Status :: 2 - Pre-Alpha',
            'Intended Audience :: Education',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: MIT License',
            'Operating System :: MacOS',
            'Operating System :: POSIX :: Linux',
            'Programming Language :: Python :: 3.6',
            'Topic :: Software Development',
            'Topic :: Scientific/Engineering',
        ],
        url='https://github.com/SYAN83/pytorch-learn',
        author='Shu Yan',
        author_email='yanshu.usc@gmail.com',
        license='MIT',
        packages=setuptools.find_packages(exclude=['tests']),
        install_requires=[
            'torch>=1.0.0',
        ],
        include_package_data=True,
        zip_safe=False
    )

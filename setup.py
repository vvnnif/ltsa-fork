from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()

setup(
    name='ltsa',
    version='0.2',
    description='Package for Local Tangent Space Alignment manfiold learning.',
    author="Charles gadd",
    author_email="cwlgadd@gmail.com",
    url="https://gitlab.com/charles1992/ltsa",
    download_url="https://gitlab.com/charles1992/ltsa",
    packages=['ltsa', 'ltsa.utils', 'ltsa.testing', ],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    long_description=readme(),
    zip_safe=False,
    test_suite='nose.collector',
    tests_require=['nose'],
)

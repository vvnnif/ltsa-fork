from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(
    name='ltsa',
    version='0.1',
    description='Package for local tangent space alignment manifold learning.',
    author= "Charles Gadd",
    author_email= "cwlgadd@gmail.com",
    packages=['ltsa', 'ltsa.utils', 'ltsa.testing', ],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    long_description=readme(),
    zip_safe=False,
    test_suite='nose.collector',
    tests_require=['nose'],
)

# ltsa

The Local Tangent Space Alignment framework in Python.

* ltsa [homepage](http://gitlab.com/charles1992/ltsa/)

<!--- * [![licence](https://img.shields.io/badge/licence-BSD-blue.svg)](http://opensource.org/licenses/BSD-3-Clause) 
-->

<!---
## Citation

    @Misc{ltsa2017,
      author =   {{ltsa}},
      title =    {{ltsa}: A Local Tangent Space Alignment framework in python},
      howpublished = {\url{http://gitlab.com/charles1992/ltsa}},
      year = {since 2017}
    }
-->

## Installing

You can install the latest release with:

    $ pip install ltsa

If you'd like to install from source, or want to contribute to the project (i.e. by sending pull requests via github)
then:

    $ git clone https://gitlab.com/charles1992/ltsa.git 
    $ cd ltsa
    $ git checkout devel
    $ python setup.py build_ext --inplace
    

## Running unit tests (development):

We use nosetests, this can be installed using pip with:

    $ pip install nose

Run nosetests from the root directory of the repository:

    $ nosetests -v ltsa/testing

or using setuptools

    $ python setup.py test




<!---
### Commit new patch to devel


A usual workflow should look like this:

    $ git fetch origin
    $ git checkout -b <pull-origin>-devel origin/<pull-origin>-devel
    $ git merge devel
    $ coverage run travis_tests.py

**Make changes for tests to cover corner cases (if statements, None arguments etc.)**
Then we are ready to make the last changes for the changelog and versioning:

    $ git commit -am "fix: Fixed tests for <pull-origin>"
    $ bumpversion patch # [optional]
    $ gitchangelog > CHANGELOG.md
    $ git commit -m "chg: pkg: CHANGELOG update" CHANGELOG.md

Now we can merge the pull request into devel:

    $ git checkout devel
    $ git merge --no-ff <pull-origin>-devel
    $ git push origin devel

This will update the devel branch of GPy.
-->

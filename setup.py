from distutils.core import setup


package_dir = {"dlmep":"dlmep"}

packages = [
      "dlmep",

]




setup(
      name="dlmep",
      version="0.1",
      description="Deep Learning Molecule Energy Predict",
      author="B.C. WANG",
      url="https://github.com/B-C-WANG",
      license="LICENSE",
      package_dir=package_dir,
      packages=packages
      )
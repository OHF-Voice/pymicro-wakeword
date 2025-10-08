from setuptools import setup, Distribution
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel


class BinaryDistribution(Distribution):
    def has_ext_modules(self):  # platform wheel
        return True


class fixed_tag_bdist_wheel(_bdist_wheel):
    def get_tag(self):
        py, abi, plat = super().get_tag()
        print(f"[bdist_wheel.get_tag] computed: python={py} abi={abi} plat={plat}")
        # Force version-agnostic tag for Python 3
        return ("py3", "none", plat)


setup(distclass=BinaryDistribution, cmdclass={"bdist_wheel": fixed_tag_bdist_wheel})

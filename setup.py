"""Builds the C++ extension:  python setup.py build_ext --inplace"""
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

setup(
    name="mcpricer_cpp",
    ext_modules=[Pybind11Extension("mcpricer_cpp", ["cpp/mc_core.cpp"], cxx_std=17)],
    cmdclass={"build_ext": build_ext},
)

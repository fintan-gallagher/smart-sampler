from setuptools import setup, find_packages

setup(
    name="smart-sampler",
    version="1.0.0",
    description="Smart audio sampler for Raspberry Pi",
    author="Fintan Gallagher",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0,<2.0.0",
        "soundfile>=0.12.0",
        "librosa>=0.10.0",
        "tensorflow>=2.13.0",
        "matplotlib>=3.7.0",
        "pyaudio>=0.2.13",
        "scipy>=1.10.0",
    ],
)

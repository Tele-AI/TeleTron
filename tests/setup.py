from setuptools import setup, find_packages

def read_readme():
    try:
        with open("README.md", "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return "To pioneer training long-context multi-modal transformer models"
    
def read_requirements():
    try:
        with open("requirements.txt", "r", encoding="utf-8") as fh:
            return [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        return []


setup(
    name="unit_test",
    version="0.1.0",
    description="To pioneer training long-context multi-modal transformer models",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="TeleAI-Infra Team",
    url="https://github.com/Tele-AI/TeleTron.git",
    install_requires=read_requirements(),
    packages=find_packages(include=['teletron*']),
    package_data={"teletron": ["**/*.yml", "**/*.sh"]},
    classifiers=[
        "Development Status :: 3 - Alpha",
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Information Technology',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Utilities',        
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="transformer, multimodal, long-context, machine learning",
    license="Apache-2.0",
    include_package_data=True,
    zip_safe=False,
    python_requires=">=3.8"
)


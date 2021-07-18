import setuptools

with open("README.md", "r",encoding='gb18030',errors='ignore') as fh:
    long_description = fh.read()

setuptools.setup(
    name="tymon",
    version="0.0.4",
    author="TymonXie",
    author_email="847250484@qq.com",
    description="An AI Assistant More Than a Toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
 install_requires=[       # 工具包的依赖包
    'scikit-learn>=0.24.2',
    'torch>=1.9.0',
     'pandas>=1.1.5',
     'numpy>=1.19',
     'matplotlib>=3.3.4',
    ],
    url="https://github.com/TymonXie/tymon",
    packages=setuptools.find_packages(),
    python_requires="~=3.6",
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
)
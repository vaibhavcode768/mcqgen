from setuptools import find_packages,setup

setup(
    name='mcqgenerator',
    version='0.0.1',
    author='vaibhav jain',
    author_email='crazygamer7659@gmail.com',
    install_requires=["ai21","langchain","streamlit","python-dotenv","PyPDF2"],
    packages=find_packages()
)
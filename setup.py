from setuptools import find_packages, setup
from typing import List

def get_requirements()-> List[str]:
    requirements_list:List[str]=[]
    try:
        with open("requirements.txt", 'r') as f:
            lines = f.readlines()
            for line in lines:
                requirements=line.strip()
                if requirements and requirements!='-e .':
                    requirements_list.append(requirements)
    except FileNotFoundError:
        print("requirement.txt not found")
    except Exception as e:
        raise e
    return requirements_list

setup(
    name="NetworkSecuriity",
    version="0.0.1",
    author="Pratham Verma",
    author_email="support@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements()
)
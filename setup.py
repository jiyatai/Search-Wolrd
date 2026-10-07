from setuptools import setup, find_packages


def load_requirements(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return [
            line.strip() for line in f
            if line.strip() and not line.startswith('#')
        ]


setup(
    name='searchworld',
    version='0.1.0',
    packages=find_packages(),
    package_dir={"": "."},
    author='SearchWorld contributors',
    author_email='',
    description='SearchWorld: spatial value-grounded imagination for UAV object search via world models',
    url='REPLACE_WITH_YOUR_REPOSITORY_URL',
    install_requires=load_requirements('requirements.txt'),
)

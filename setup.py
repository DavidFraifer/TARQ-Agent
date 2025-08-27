from setuptools import setup, find_packages

setup(
    name='tarq_agent',
    version='0.3.0',
    description='TARQ (Temporal Async Reasoning & Queueing)',
    author='David Serrano Díaz',
    author_email='davidsd.2704@gmail.com',
    packages=find_packages(),
    install_requires=[],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License ::  CC BY-NC 4.0',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)

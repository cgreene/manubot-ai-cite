import setuptools

# Commands to publish new package:
#
# rm -rf dist/
# python setup.py sdist
# twine upload dist/*

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="manubot-ai-cite",
    version="0.0.1",
    author="Casey Greene",
    author_email="Casey.S.Greene@cuanschutz.edu",
    description="A Manubot plugin to suggest revisions using the OpenAI API",
    license="BSD-2-Clause Plus Patent",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/chpdm/manubot-ai-cite",
    package_dir={"": "libs"},
    packages=[
        "manubot_ai_cite/",
    ],
    python_requires=">=3.10",
    install_requires=[
        "openai>=0.27",
        "pyyaml",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
)

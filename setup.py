import setuptools

setuptools.setup(
        name = 'ponaxt',
        packages = setuptools.find_packages(),
        install_requires=[
            'numpy', 'torch', 'tqdm'],
        entry_points = {
            'console_scripts':['ponaxt = ponaxt.cli.ponaxt:main']})


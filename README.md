# COSMIC: Characterization Of Star clusters using Machine-learning Inference and Clustering

## Overview
COSMIC is an open-source software suite built on Python 3.11 and the PyMC library. It employs unsupervised machine learning techniques and Bayesian estimation to analyze extensive datasets from the Gaia satellite, focusing on open star clusters. The software aims to accurately identify key parameters such as structural attributes, reddening, age, and distance.

**Note**: This project is under heavy development and is not recommended for scientific work at this time.

## Contributors
- MSc. Lucas Pulgar-Escobar, Universidad de Concepción, Chile
- Nicolás Henríquez Salgado, Universidad de Concepción, Chile

## License
COSMIC is licensed under the AGPL-3.0 License. Please see the [LICENSE.md](LICENSE.md) file for more details.

## Installation

### Editable (developer) install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The editable install keeps your virtual environment in sync with the repository, so any code changes are reflected immediately without reinstalling the package. The `dev` extra brings in the lightweight tooling needed to run tests.

### Legacy requirements file

If you prefer the older workflow, the pinned dependencies are still available:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Usage
_Coming Soon: A quick start guide and usage examples._

## Dependencies
COSMIC relies on a variety of Python packages for its functionality. For Bayesian modeling, we use the [PyMC](https://github.com/pymc-devs/pymc) library. Clustering capabilities are powered by [HDBSCAN](https://github.com/scikit-learn-contrib/hdbscan). Data visualization and statistical graphics are implemented using `matplotlib` and `seaborn`. For astronomical data handling, we utilize `astropy`.

## Acknowledgments
We would like to extend our gratitude to the developers behind [PyMC](https://github.com/pymc-devs/pymc) and [HDBSCAN](https://github.com/scikit-learn-contrib/hdbscan) for their invaluable libraries, which form the backbone of COSMIC's Bayesian and clustering functionalities, respectively.

## Contributing
Interested in contributing? Please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines.

## Support and Contact
For questions or support, please [open an issue](https://github.com/yourusername/COSMIC/issues) on our GitHub page. Alternatively, you can contact Lucas Pulgar-Escobar at lescobar2019@udec.cl.

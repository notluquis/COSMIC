# COSMIC: Characterization Of Star clusters using Machine-learning Inference and Clustering

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Development Status](https://img.shields.io/badge/status-alpha-red.svg)](https://github.com/notluquis/COSMIC)

COSMIC is an open-source Python package for analyzing star clusters using machine learning techniques and Bayesian inference. Built specifically for processing Gaia satellite data, COSMIC employs unsupervised clustering algorithms and statistical analysis to identify and characterize open star clusters.

## ⚠️ Development Status

**This project is currently in alpha development (v0.0.1) and is not yet recommended for production scientific work.** The API may change significantly between versions as we work toward a stable release.

## 🚀 Features

- **Advanced Clustering**: HDBSCAN-based clustering with hyperparameter optimization via Optuna
- **Gaia Integration**: Native support for Gaia data formats and photometric systems
- **Flexible Preprocessing**: Comprehensive data cleaning and preparation tools
- **Analysis Tools**: Statistical analysis and visualization of cluster properties
- **Modular Design**: Clean, extensible architecture with backward-compatible shims

## 📦 Installation

### From Source (Recommended for v0.0.1)

```bash
# Clone the repository
git clone https://github.com/notluquis/COSMIC.git
cd COSMIC

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install --upgrade pip
pip install -e ".[dev]"
```

### Requirements

- Python 3.11 or higher
- See `pyproject.toml` for complete dependency list

## 🔧 Quick Start

```python
import cosmic

# Load Gaia data
loader = cosmic.DataLoader("your_gaia_catalog.ecsv")
data = loader.load_data(systems=["Gaia", "TMASS"])

# Preprocess the data
preprocessor = cosmic.DataPreprocessor(data)
good_data, bad_data = preprocessor.process()

# Perform clustering
clusterer = cosmic.Clustering(good_data, bad_data)
clusterer.search(['pmra', 'pmdec', 'parallax'])

# Analyze results
analyzer = cosmic.ClusterAnalyzer(clusterer.combined_data)
analyzer.run_analysis()
```

## 📁 Project Structure

```
cosmic/
├── core/           # Clustering algorithms and core functionality
├── io/             # Data loading and I/O utilities  
├── preprocess/     # Data preprocessing and cleaning
├── analysis/       # Statistical analysis and characterization
└── utils/          # General utility functions
```

## 🔬 Core Components

### Clustering (`cosmic.Clustering`)
- HDBSCAN-based clustering with persistence thresholding
- Grid search and Optuna optimization for hyperparameters
- Multiple validation metrics (relative validity, DBCV, cluster persistence)

### Data Loading (`cosmic.DataLoader`)
- Support for multiple photometric systems (Gaia, 2MASS, WISE)
- Flexible column mapping and data validation
- Built-in quality filtering

### Preprocessing (`cosmic.DataPreprocessor`)
- Zero-point corrections for photometry
- Proper motion corrections
- Missing value handling and outlier detection

### Analysis (`cosmic.ClusterAnalyzer`)
- Statistical characterization of clusters
- Integration with external tools (Sagitta for stellar parameters)
- Comprehensive plotting and visualization

## 🤝 Contributing

We welcome contributions! Please see our [contribution guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/notluquis/COSMIC.git
cd COSMIC
pip install -e ".[dev]"
pytest  # Run tests
```

## 📖 Documentation

- [API Reference](https://github.com/notluquis/COSMIC/wiki) (Coming Soon)
- [Examples](notebooks/) - Jupyter notebooks with usage examples
- [Changelog](CHANGELOG.md) - Version history and changes

## 📄 License

COSMIC is licensed under the [GNU Affero General Public License v3.0](LICENSE). This ensures that any modifications or derivative works remain open source.

## 👥 Authors

- **Lucas Pulgar-Escobar** - Universidad de Concepción, Chile ([lescobar2019@udec.cl](mailto:lescobar2019@udec.cl))
- **Nicolás Henríquez Salgado** - Universidad de Concepción, Chile

## 🙏 Acknowledgments

COSMIC builds upon excellent open-source libraries:
- [HDBSCAN](https://github.com/scikit-learn-contrib/hdbscan) for clustering algorithms
- [Optuna](https://optuna.org/) for hyperparameter optimization  
- [Astropy](https://www.astropy.org/) for astronomical data handling
- [scikit-learn](https://scikit-learn.org/) for machine learning utilities

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/notluquis/COSMIC/issues)
- **Email**: [lescobar2019@udec.cl](mailto:lescobar2019@udec.cl)
- **Discussions**: [GitHub Discussions](https://github.com/notluquis/COSMIC/discussions)

## 🗺️ Roadmap

- [ ] v0.1.0: Stable API and comprehensive documentation
- [ ] v0.2.0: PyPI distribution and additional clustering algorithms
- [ ] v1.0.0: Production-ready release with full validation suite

---

**Citation**: If you use COSMIC in your research, please cite our paper (in preparation) and acknowledge the underlying libraries.

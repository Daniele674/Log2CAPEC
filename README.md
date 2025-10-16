# Log2CAPEC

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
![Status](https://img.shields.io/badge/status-research_prototype-orange)

An automated end-to-end pipeline to translate raw, heterogeneous honeypot logs into standardized MITRE CAPEC attack patterns. Log2CAPEC leverages a Large Language Model for semantic interpretation and a hybrid retrieval engine (semantic + keyword with Reciprocal Rank Fusion) for accurate, context-aware mapping of cyber threats.

---

## 📖 Table of Contents

- [Introduction: The Problem](#-introduction-the-problem)
- [The Solution: Log2CAPEC](#-the-solution-log2capec)
- [✨ Key Features](#-key-features)
- [🏗️ System Architecture](#️-system-architecture)
- [🛠️ Tech Stack](#️-tech-stack)
- [🚀 Installation and Setup](#-installation-and-setup)
- [⚙️ Usage](#️-usage)
- [📂 Project Structure](#-project-structure)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)
- [✍️ How to Cite](#️-how-to-cite)

## 📌 Introduction: The Problem

Security log analysis is critical for threat intelligence, but security teams are often overwhelmed by a "data deluge." Multi-honeypot platforms like **TPOT** generate a massive volume of raw and heterogeneous data (SSH session logs, network scans, emulated service interactions, etc.), making manual analysis impractical.

This project addresses three core challenges:
1.  **Data Heterogeneity:** How to uniformly analyze logs from diverse sources (Cowrie, Honeytrap, Dionaea, etc.)?
2.  **Lack of Context:** How to reconstruct an attacker's intent from a series of atomic and often obfuscated events?
3.  **Need for Standardization:** How to translate raw observations into structured, actionable knowledge using a granular framework like **MITRE CAPEC**?

## 💡 The Solution: Log2CAPEC

**Log2CAPEC** is an end-to-end pipeline that automates the interpretation and mapping of honeypot logs. Its innovative approach is based on three key methodological contributions:

1.  **Holistic Actor-Centric Aggregation:** Instead of analyzing isolated logs, the system aggregates all events related to a single actor (by IP or session ID), reconstructing their **complete behavioral profile**.
2.  ***Upstream* Semantic Interpretation:** A Large Language Model (LLM) acts as an "interpreter" early in the pipeline. Using advanced prompt engineering (combining role-playing, chain-of-thought, and few-shot learning), the LLM translates the attacker's behavior into a structured JSON analysis, which serves as an enriched query.
3.  **Hybrid Matching with Rank Fusion:** A hybrid search engine matches the LLM's query against a CAPEC knowledge base. It employs **Reciprocal Rank Fusion (RRF)** to combine the results from a semantic search (using `ATT&CK-BERT` embeddings) and a lexical search (using `TF-IDF`), ensuring a robust and accurate final ranking.

## ✨ Key Features

- **Heterogeneous Log Analysis:** Built-in support for major TPOT honeypots, including Cowrie, Honeytrap, Dionaea, SentryPeer, and CiscoASA.
- **Advanced Semantic Interpretation:** Utilizes `Mistral-7B-Instruct-v0.2` with contextual prompt engineering for high-level analysis.
- **Hybrid Search Engine:** Combines the contextual understanding of `ATT&CK-BERT` with the term precision of `TF-IDF`.
- **Advanced Rank Fusion:** Employs Reciprocal Rank Fusion (RRF) for a methodologically sound ranking combination.
- **Resource-Efficient:** Runs large LLMs on accessible hardware via 4-bit quantization.
- **Interpretable:** The output includes the LLM's justification and a debug table of the fusion process for full transparency.
- **Open Source:** Entirely built on open-source technologies and released under the MIT License.

## 🏗️ System Architecture

The architecture is a modular pipeline that transforms raw data into structured knowledge.

<img width="1776" height="2456" alt="diagramma" src="https://github.com/user-attachments/assets/a4f49695-2f46-4371-ba63-889884fa8c63" />

## 🛠️ Tech Stack

This project is developed in **Python 3.10+** and relies on a rich ecosystem of open-source libraries.

- **Data Manipulation:**
  - `pandas`: For reading and handling the CSV log dataset.
- **Linguistic Preprocessing (for CAPEC):**
  - `nltk`, `spacy`: For lemmatization, POS tagging, and stopword removal.
- **Large Language Model:**
  - `torch`, `transformers` (Hugging Face): For loading and running the model.
  - `bitsandbytes`, `accelerate`: For 4-bit quantization of the `mistralai/Mistral-7B-Instruct-v0.2` model.
- **Information Retrieval (Hybrid Matching):**
  - `sentence-transformers`: To generate embeddings with the `basel/ATT&CK-BERT` model.
  - `chromadb`: As the vector database for efficient semantic search.
  - `scikit-learn`: For the TF-IDF model implementation.

## 🚀 Installation and Setup

Follow these steps to set up the environment and run the project.

**1. Clone the Repository**
```bash
git clone https://github.com/your-username/Log2CAPEC.git
cd Log2CAPEC
```

**2. Create a Virtual Environment (Recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate```

**3. Install Dependencies**
This project requires several large libraries. Ensure your `pip` is up to date.
```bash
pip install -r requirements.txt
```

**4. Download NLP Models**
The first time you run the script, it will automatically download the necessary models for `spacy` and `nltk`.

**5. Prepare the Data**
- Place the MITRE CAPEC catalog file in the project root and name it `CAPEC.xml`.
- Place your honeypot log dataset in the project root and ensure its name matches the `HONEYPOT_CSV_FILE` variable in the script (e.g., `tpot_less.csv`).

## ⚙️ Usage

To run the full analysis pipeline, execute the main script from the project root:

```bash
Log2CAPEC.py
```

The script will perform the following steps:
1.  Initialize the system, loading the LLM and building the CAPEC knowledge base (this may take longer on the first run).
2.  Read and preprocess the log CSV file, aggregating events by actor.
3.  Iterate through each behavioral profile, generating the LLM analysis and the hybrid mapping.
4.  Print the detailed results for each analyzed session/actor to the console, including the RRF debug table.

## 📂 Project Structure

```
Log2CAPEC/
│
├── Log2CAPEC.py      # Main script to run the entire pipeline
├── requirements.txt         # Python dependency list
├── CAPEC.xml                # MITRE CAPEC catalog file
├── tpot_less.csv            # Example honeypot log dataset
├── LICENSE                  # Project license file (MIT)
└── README.md                # This file
```

## 🤝 Contributing

Contributions are welcome! If you have ideas for improvements, bug fixes, or new features, please feel free to:
1.  Open an "Issue" to discuss your idea.
2.  Fork the repository and create a "Pull Request" with your changes.

## 📜 License

This project is released under the **MIT License**. See the `LICENSE` file for more details.

## ✍️ How to Cite

If you use this code or methodology in your research, please cite the following thesis:

```bibtex
@mastersthesis{Gregori2025Log2CAPEC,
  author    = {Daniele Gregori},
  title     = {Semantic Analysis and Hybrid Mapping of Honeypot Logs to CAPEC Attack Patterns via Large Language Models},
  school    = {University of Salerno},
  year      = {2025},
  month     = {MM}, % Replace with the month of defense
  type      = {Master's Thesis in Computer Science}
}
```

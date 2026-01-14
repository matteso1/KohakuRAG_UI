# KohakuRAG UI - WattBot Chatbot Interface

> **Research Cyberinfrastructure Exploration Project**  
> Building a RAG-powered chatbot for sustainable AI research using the WattBot corpus

## 🎯 Project Overview

This project builds a user-friendly chatbot interface for the [KohakuRAG](https://github.com/KohakuBlueleaf/KohakuRAG) pipeline—the #1 solution from the 2025 WattBot Challenge. The chatbot answers questions about environmental impacts of AI using a curated corpus of energy and sustainability research papers.

### Team

| Member | Role | Branch |
|--------|------|--------|
| **Chris** | Research Supervisor | - |
| **Blaise** | Local/On-Prem Development | `local` |
| **Nils** | AWS Bedrock Integration | `bedrock` |

### Goals

- ✅ Create a Streamlit-based chat interface for non-technical users
- ✅ Deploy on AWS using Bedrock for on-demand LLM access (low idle cost)
- ✅ Compare with self-hosted options (GB10, campus GPU cluster)
- ✅ Serve as a reference deployment for future RCI consultations

---

## 📁 Repository Structure

```
KohakuRAG_UI/
├── README.md                 # This file
├── docs/
│   └── bedrock-integration-proposal.md   # Nils' Bedrock design proposal
├── src/                      # (Future) Streamlit app code
├── configs/                  # (Future) Configuration files
└── .env.example             # Environment variable template
```

---

## 🔗 Related Resources

- **KohakuRAG Core**: [github.com/KohakuBlueleaf/KohakuRAG](https://github.com/KohakuBlueleaf/KohakuRAG)
- **WattBot Competition**: [Kaggle WattTime Challenge](https://www.kaggle.com/competitions/wattbot-2025)
- **AWS Console**: [uw-madison-dlt3.awsapps.com](https://uw-madison-dlt3.awsapps.com/start/#/console?account_id=183295408236&role_name=ml-bedrock-183295408236sagemaker)

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- AWS CLI (for Bedrock branch)
- Access to the WattBot corpus and KohakuRAG artifacts

### Quick Start

```bash
# Clone the repository
git clone https://github.com/matteso1/KohakuRAG_UI.git
cd KohakuRAG_UI

# (Coming soon) Install dependencies
pip install -r requirements.txt

# (Coming soon) Run the Streamlit app
streamlit run src/app.py
```

---

## 📋 Branch Overview

### `bedrock` Branch (Nils)

**Status**: 🟡 Planning Phase

The Bedrock branch integrates AWS Bedrock for managed LLM inference. See [docs/bedrock-integration-proposal.md](docs/bedrock-integration-proposal.md) for the full design proposal.

### `local` Branch (Blaise)

**Status**: 🟡 In Development

The local branch uses a small local LLM (< 1B parameters) for development and on-prem deployment.

---

## 📝 Meeting Notes

### Week 1 (1/13/2026)

- Introductions completed
- Tasks assigned:
  - **Nils**: Begin planning Bedrock integration, create workflow diagram
  - **Blaise**: Get local chatbot working in Streamlit
  - **Chris**: Provide AWS Bedrock access

---

## 📄 License

This project is part of the UW-Madison Research Cyberinfrastructure Exploration initiative.

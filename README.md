# WattBot RAG User Interface

A Streamlit-based chat interface for the KohakuRAG pipeline, enabling interactive Q&A over the WattBot research corpus on sustainable AI.

## Project Context

This repository supports the Research Cyberinfrastructure Exploration initiative at UW-Madison. The goal is to build a long-running chatbot that answers questions about the environmental impacts of AI using a curated corpus of energy and sustainability research papers.

The project uses [KohakuRAG](https://github.com/KohakuBlueleaf/KohakuRAG), the top-ranked solution from the 2025 WattBot Challenge, as the core retrieval engine. This repository focuses on:

1. Building a user-facing Streamlit interface
2. Deploying the system on AWS using Bedrock for LLM inference
3. Comparing managed cloud deployment against self-hosted alternatives

## Architecture

The system follows a standard RAG (Retrieval-Augmented Generation) architecture:

```mermaid
flowchart TB
    U[User Query] --> S[Streamlit UI]
    S --> P[RAG Pipeline]
    P --> E[Jina Embeddings]
    P --> V[(Vector Store<br/>SQLite)]
    P --> L[LLM Backend]
    L --> B[AWS Bedrock]
    L --> Local[Local Models]
```

### Deployment Options

| Approach | LLM Backend | Use Case |
|----------|-------------|----------|
| AWS Bedrock | Managed foundation models via API | Production, on-demand usage |
| Local | Small models under 1B params | Development, on-prem deployment |

## Documentation

- [Bedrock Integration Proposal](docs/bedrock-integration-proposal.md) - AWS Bedrock design and implementation plan
- [Meeting Notes](docs/meeting-notes.md) - Team discussions and decisions

## Repository Structure

```
.
├── docs/
│   ├── bedrock-integration-proposal.md   # AWS Bedrock design document
│   └── meeting-notes.md                  # Team meeting notes
├── src/                                  # Application source code
├── vendor/KohakuRAG/configs/              # Pipeline & experiment configs
├── .env.example                          # Environment template
└── README.md
```

## Development Branches

- **main**: Stable releases and documentation
- **bedrock**: AWS Bedrock integration (Nils)
- **local**: Local/on-prem LLM support (Blaise)

## Related Resources

- [KohakuRAG](https://github.com/KohakuBlueleaf/KohakuRAG) - Core RAG engine
- [WattBot 2025 Competition](https://www.kaggle.com/competitions/WattBot2025/overview) - Original challenge
- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/) - Managed LLM service
- [Generative AI with Amazon Bedrock](https://www.coursera.org/learn/generative-ai-applications-amazon-bedrock) - Coursera course

## Team

| Name | Role | GitHub |
|------|------|--------|
| Chris Endemann | Research Supervisor | [@qualiaMachine](https://github.com/qualiaMachine) |
| Blaise Enuh | Local deployment | [@EnuhBlaise](https://github.com/EnuhBlaise) |
| Nils Matteson | AWS Bedrock integration | [@matteso1](https://github.com/matteso1) |

## License

Research project under UW-Madison Research Cyberinfrastructure.

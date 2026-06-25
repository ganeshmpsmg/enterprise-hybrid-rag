# MkDocs Configuration
site_name: Enterprise Hybrid RAG System
site_url: https://enterprise-rag.readthedocs.io
site_author: ML Engineering Team
site_description: Production-grade RAG system documentation

repo_url: https://github.com/yourusername/enterprise-hybrid-rag
repo_name: enterprise-hybrid-rag

theme:
  name: material
  features:
    - navigation.tabs
    - navigation.sections
    - navigation.expand
    - search.suggest
    - content.code.copy
  palette:
    - scheme: default
      primary: indigo
      accent: deep purple

nav:
  - Home: index.md
  - Architecture: architecture/README.md
  - Guides:
    - Deployment: guides/DEPLOYMENT_GUIDE.md
    - Production: guides/PRODUCTION_BEST_PRACTICES.md
  - API Reference: api/README.md
  - Interview Q&A: guides/INTERVIEW_QA.md

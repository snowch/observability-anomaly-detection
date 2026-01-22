# Embedding-Based Anomaly Detection for Observability
 
A comprehensive tutorial series on building production-ready anomaly detection systems using ResNet embeddings for OCSF (Open Cybersecurity Schema Framework) observability data.
 
## 📖 Read the Tutorial
 
👉 **[Start the tutorial series](https://snowch.github.io/observability-anomaly-detection/)**
 
## What You'll Learn
 
How to build, train, and deploy a **custom embedding model** (TabularResNet) specifically designed for OCSF observability data:
 
- Build and train TabularResNet using self-supervised learning on unlabeled logs
- Deploy the model as a FastAPI service for real-time inference
- Store embeddings in a vector database for fast k-NN similarity search
- Detect anomalies through vector operations (no separate detection model needed)
- Monitor embedding quality and trigger automated retraining
 
## Tutorial Series
 
1. **Part 1**: Understanding ResNet Architecture
2. **Part 2**: Adapting ResNet for Tabular Data
3. **Part 3**: Feature Engineering for OCSF Data
4. **Part 4**: Self-Supervised Training
5. **Part 5**: Evaluating Embedding Quality
6. **Part 6**: Anomaly Detection Methods
7. **Part 7**: Production Deployment
8. **Part 8**: Production Monitoring
9. **Part 9**: Multi-Source Event Correlation
 
**Plus**: Hands-on Jupyter notebooks and sample data
 
## Who This Is For
 
- ML engineers building anomaly detection systems
- Security engineers working with observability data
- Data scientists interested in self-supervised learning
- Anyone wanting to apply ResNet to tabular/observability data
 
## Prerequisites
 
- Basic Python and PyTorch
- Understanding of neural networks (or see our [Neural Networks From Scratch](https://snowch.github.io/ai-eng/nnfs/) series)
 
## Quick Start
 
### Run the Hands-on Notebooks
 
1. Install dependencies:
```bash
pip install pandas numpy torch scikit-learn matplotlib pyarrow
```
 
2. Download sample data and notebooks from the [Appendix](https://snowch.github.io/observability-anomaly-detection/appendix-notebooks)
 
3. Run the notebooks:
   - `03-feature-engineering.md` - Extract features from OCSF data
   - `04-self-supervised-training.md` - Train TabularResNet
   - `05-embedding-evaluation.md` - Evaluate embedding quality
   - `06-anomaly-detection.md` - Detect anomalies
 
### Generate Your Own Data
 
Use the [Docker Compose stack](https://snowch.github.io/observability-anomaly-detection/appendix-generating-training-data) to generate realistic OCSF observability data with labeled anomalies.
 
## Key Features
 
✅ **Production-ready code** - All examples are deployable
✅ **No labels required** - Self-supervised learning on unlabeled data
✅ **Hands-on notebooks** - Working code you can run immediately
✅ **Sample data included** - Pre-generated OCSF events
✅ **Complete MLOps** - Deployment, monitoring, retraining
 
## Applicability Beyond OCSF
 
While this series uses OCSF security logs as the running example, the TabularResNet approach applies to **any structured observability data**:
 
- **Telemetry/Metrics**: CPU%, memory, latency with metadata
- **Configuration data**: Key-value pairs, settings
- **Distributed traces**: Span attributes
- **Application logs**: JSON logs, syslog
 
**The key requirement**: Your data can be represented as rows with categorical and numerical features.
 
## License
 
This tutorial series is licensed under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).
 
## Author
 
**Chris Snow** - [snowch.github.io](https://snowch.github.io)
 
## Related Resources
 
- [Embeddings at Scale Book](https://snowch.github.io/embeddings-at-scale-book/) - Deep dive into production embedding systems
- [Neural Networks From Scratch](https://snowch.github.io/ai-eng/nnfs/) - Learn NN fundamentals
 
## Contributing
 
Found an issue or have suggestions? [Open an issue](https://github.com/snowch/observability-anomaly-detection/issues) or submit a pull request!

# Regularization in Linear Regression

Examples that build on [`../variance/README.md`](../variance/README.md) and
study how regularization pulls back the parameter variance that noisy
training data introduces, by fixing the noise level and instead sweeping the
regularization strength.

| Example | Studies | Details |
|---|---|---|
| [`L2/`](./L2) | L2 (Ridge) weight decay — independent `weights_decay` and `bias_decay` terms | [README](./L2/README.md) |
| [`DL2/`](./DL2) | Isolating decay's effect from noise on a fixed dataset, across repeated trials *(work in progress — only the no-decay baseline is implemented)* | [README](./DL2/README.md) |

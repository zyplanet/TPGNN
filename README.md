# TPGNN

## Multivariate Time-Series Forecasting with Temporal Polynomial Graph Neural Networks (****NeurIPS 2022****)
[![Teaser.png](https://s1.ax1x.com/2023/06/14/pCng98f.png)](https://imgse.com/i/pCng98f)

Modeling multivariate time series (MTS) is critical in modern intelligent systems. The accurate forecast of MTS data is still challenging due to the complicated latent variable correlation. Recent works apply the Graph Neural Networks (GNNs) to the task, with the basic idea of representing the correlation as a static graph. 

However, predicting with a static graph causes significant bias because the correlation is time-varying in the real-world MTS data. Besides, there is no gap analysis between the actual correlation and the learned one in their works to validate the effectiveness. 

This paper proposes a temporal polynomial graph neural network (TPGNN) for accurate MTS forecasting, which represents the dynamic variable correlation as a temporal matrix polynomial in two steps. First, we capture the overall correlation with a static matrix basis. Then, we use a set of time-varying coefficients and the matrix basis to construct a matrix polynomial for each time step.

### Usage

Installing dependency

```bash
pip install -r requirements.txt
```

Generating time stamp for MTS data

```bash
sh ./scripts/genstamp.sh
```

Starting Training Process

```bash
sh ./scripts/multi.sh
```

Check these scripts to configure parameters if necessary.

More parameter configurations can be found in **config.py**.

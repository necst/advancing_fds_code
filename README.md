# Fraud Detection Framework

This repository contains the source code and additional material related to the 
paper *"Advancing Fraud Detection Systems through Online Learning"* [[1]](#references) [URL](https://link.springer.com/chapter/10.1007/978-3-031-43427-3_17).

>[!IMPORTANT]
> This code requires a banking dataset that, for privacy reasons, we cannot share with the scientific community.
> The dataset is characterized by the following features: Amount, IBAN, IBAN\_CC, 

>[!NOTE]
> If you use this code, please include the following citation:
```
@inproceedings{paladini2023advancingfds,
  author       = {Tommaso Paladini and
                  Martino {Bernasconi de Luca} and
                  Michele Carminati and
                  Mario Polino and
                  Francesco Trov{\`{o}} and
                  Stefano Zanero},
  title        = {{Advancing Fraud Detection Systems Through Online Learning}},
  booktitle    = {Machine Learning and Knowledge Discovery in Databases: Applied Data
                  Science and Demo Track - European Conference, {ECML} {PKDD} 2023,
                  Turin, Italy, September 18-22, 2023, Proceedings, Part {VI}},
  series       = {Lecture Notes in Computer Science},
  volume       = {14174},
  pages        = {275--292},
  publisher    = {Springer},
  year         = {2023},
  url          = {https://doi.org/10.1007/978-3-031-43427-3\_17},
  doi          = {10.1007/978-3-031-43427-3\_17},
  timestamp    = {Sun, 12 Nov 2023 02:12:57 +0100},
}
```

## Abstract
The rapid increase in digital transactions has led to a consequential surge in financial fraud, requiring an automatic way of defending effectively from such a threat. The past few years experienced a rise in the design and use by financial institutions of different machine learning-based fraud detection systems. However, these solutions may suffer severe drawbacks if a malevolent adversary adapts their behavior over time, making the selection of the existing fraud detectors difficult. In this paper, we study the application of online learning techniques to respond effectively to adaptive attackers. More specifically, the proposed approach takes as input a set of classifiers employed for fraud detection tasks and selects, based on the performances experienced in the past, the one to apply to analyze the next transaction. The use of an online learning approach guarantees to keep at a pace the loss due to the adaptive behavior of the attacker over a given learning period. To validate our methodology, we perform an extensive experimental evaluation using real-world banking data augmented with distinct fraudulent campaigns based on real-world attackers’ models. Our results demonstrate that the proposed approach allows prompt updates to detection models as new patterns and behaviors are occurring, leading to a more robust and effective fraud detection system.

## Usage 

### Setup
This repository requires `Python >=3.9`. Create a virtual environment with `$ python3.9 -m venv .venv`, activate the environment with `$ source .venv/bin/activate` and install requirements with `$ pip install -r requirements.txt`.

### Dataset Generation
[!NOTE]
> Differently than the paper, in the code, fraud strategies have different names:  
> SHORT\_TERM (CODE) --> MEDIUM\_TERM (PAPER)  
> SINGLE\_FRAUD (CODE) --> SHORT\_TERM (PAPER)    
> LONG\_TERM (CODE) --> LONG\_TERM (PAPER)   

To create a synthetic fraud campaign, first create a configuration file with: 
```bash
$ mkdir config/dataset_synth_fraud/campaigns
$ touch config/dataset_synth_fraud/campaigns/CAMPAIGN_NAME.json  # e.g., lt_is.json
```

Example of configuration:
```json
{
    "campaign_name": "lt_is",
    "victims_number_2014_15": 300,
    "strategies": [
        {
            "percentage": 100,
            "min_frauds": 28,
            "max_frauds": 112,
            "min_amount": 200,
            "max_amount": 1000,
            "total_amount": 50000,
            "duration_h": 672,
            "strategy": "rand_gauss",
            "hijack": false
        }
    ]
}
```

Then, to generate the synthetic fraud run the following:
```bash
$ python -m synth_fraud_crafting.craft_campaign_all_year DATASET CAMPAIGN_NAME
```

Where `DATASET` points to the banking data, while `CAMPAIGN_NAME` is the path of the configuration file manually included in the previous step.

Then, generate training and test data with `synth_fraud_crafting.craft_campaign_weekly`. Please note that names are hardcoded, but can be extended. The script will grab data from the campaign generated for the whole "data duration", and mix them *weekly* according to a string. For more information, check the related code.

For example, create a (LT,IS) training set with:
```bash
python3 -m synth_fraud_crafting.craft_campaign_weekly llllll0 OBSERVED_CAMPAIGN_NAME_TRAIN
```

and a test set with the alternating (LT,IS) and (ST,TH) with:
```bash
$ python3 -m synth_fraud_crafting.craft_campaign_weekly 000000lFlFlFlflFlFlF OBSERVED_CAMPAIGN_NAME_TEST
```

Lastly, "aggregate" data to bring the dataset in feature space with:
```bash
python3 -m preprocessing.aggregation DATASET dataset/fraud_generated/OBSERVED_CAMPAIGN_NAME
```

### Model Selection
To train and select hyperparameters and feature sets of the FDSs, refer to the code in `model_selection`.

### Experiments
Experimental code can be found in `experiments/exp1.py`, `experiments/exp2.py`, `experiments/MWU.ipynb`. 

## References
[1] `Tommaso Paladini, Martino Bernasconi de Luca, Michele Carminati, Mario Polino, Francesco Trovò, and Stefano Zanero. "Advancing Fraud Detection Systems Through Online Learning." In Joint European Conference on Machine Learning and Knowledge Discovery in Databases, pp. 275-292. Cham: Springer Nature Switzerland, 2023.`
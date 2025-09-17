This repository contains the implementation of ConCorD, a methodology for addressing content obstruction in multimodal models.

## Installation

1. Modify the transformers library, see generation/utils.py for the changes.

2. The implementation requires modifying the forward method of target models. The modification follows a similar pattern to what's done for qwen2.5vl in the transformers library.

## Data
CoCo 2014 dataset is used for evaluation. The dataset should be downloaded and placed in the appropriate directory as specified in the scripts. In additon, you can also find other datasets in huggingface according to the name of the data in folder.

## Core Logic

The core logic of ConCord is implemented in `transformers/generation/utils.py`. This handles the decoding under content obstruction for multimodal models.

## Usage

### Reproducing Chair Metrics
To reproduce the Chair metrics evaluation:
```bash
./run_qwen_coco.sh
```

### Testing and Metrics Output
To test the model and generate Chair metrics:
```bash
./run_evaluate.sh
```

## Model Modification

When using ConCorD with a model, you need to modify the model's forward method. Please refer to the implementation of qwen2.5vl in the transformers library as an example of how to make these modifications.

## References

For more details on the methodology and evaluation metrics, please refer to the associated paper.

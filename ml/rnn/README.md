
Create a conda enviroment.

```bash
conda env create -f enviroment.yaml

conda activate rnn
```

Prepare data.

```bash
python src/prepare_data.py
```

Train model.

```bash
python src/train.py
```

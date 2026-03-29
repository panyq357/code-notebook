
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

Run prediction app.

```bash
python3 src/predict.py
```

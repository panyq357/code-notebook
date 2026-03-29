Download Data from [ChineseNlpCorpus](https://github.com/SophonPlus/ChineseNlpCorpus)

```bash
mkdir -p data/raw
cd data/raw
wget "https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/refs/heads/master/datasets/online_shopping_10_cats/online_shopping_10_cats.zip"
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

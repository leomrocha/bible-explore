# bible-explore
Exploring Bible datasets, mainly from Kaggle

This repository contains a simple search and display fo the [Kaggle Bible Corpus](https://www.kaggle.com/oswinrh/bible).

My intention with this experiment is to study ways of exploring a text dataset.

## Development

0. Clone the repo 
`git clone git@github.com:leomrocha/bible-explore.git`

1. Install the dependencies
```bash
pip3 install -r requirements.txt
```
2. Download the Kaggle Bible Corpus Dataset

From [here](https://www.kaggle.com/oswinrh/bible)

And select the language you want (this demo is built with the english one but it can be changed)

3. Encode the dataset and compute similarities 

Even if this description is not complete, there is a notebook that allows to encode and explore everything in the `notebooks/bible-explore-one.ipynb` directory

You should have 3 python pickled files as output in a `db` directory:
```
db/bible-db.pkl
db/bible-embeddings.pkl
db/graph-db.pkl
```

4. launch the development server
```bash
uvicorn src.server:app --reload
```

5. Develop
And you can create a _Pull Request_ if you make something :)
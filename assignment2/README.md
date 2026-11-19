# Assignment 2

[Written solutions](assignment2.pdf) cover Word2Vec, Adam, dropout, and dependency parsing. The LaTeX source is in `latex/`. Build it with `xelatex main.tex` twice from that directory.

The dependency parser achieved 88.65% dev UAS and 89.09% test UAS. To reproduce the run, extract the `data/` folder from the [Stanford CS224N Winter 2026 starter ZIP](https://web.stanford.edu/class/cs224n/assignments_w26/a2.zip) into this directory. With Python 3.12, run the following from this directory.

```sh
python -m pip install -r requirements-local.txt
python parser_transitions.py part_c
python parser_transitions.py part_d
python parser_model.py -e -f
python run.py
```

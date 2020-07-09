# CBMOS

This repository holds our code for studying center-based models from a numerical point of view. The figures in our manuscript can be reproduced by running the different jupyter notebooks.

## Usage

The notebook `basic-example.ipynb` describes how to simulate a small cell sheet
and plot the result. All the other notebooks describe how we generated each
plot in our manuscript, whose preprint is available on [bioRXiv](https://www.biorxiv.org/content/10.1101/2020.03.16.993246v1.abstract).

## Installing dependencies

Software dependencies are listed in the file `requirements.txt` and can be
installed easily using `pip`:

```
pip install -r requirements.txt
```

## Testing

Extensive tests have been written during the development of this software and
can be run in one command:

```
python -m pytest
```

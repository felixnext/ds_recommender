# Data Science Recommender System

This is a recommender system that will output data science resources based on user information. Data is based on the IBM Watson Platform.

> This is part of a udacity task.

## Getting Started

In order to run the system follow these steps:

```
pip install .
```

Code samples can be found in the `notebooks` sections

## Additional Remarks

The code regarding the actual recommender system can be found in the `notebooks` folder. It contains a compiled `html` version as well as a notebook with code filled in. The notebook relies on the `ask_watson` package, however does not require you to install it.

Note: Part IV of the notebook creates a custom recommender system that uses the glove embeddings. The code contains a script to automatically download the required modules, this might take additional disk space! (~3GB).

## ToDo List

* Based on interest the `ask_watson` module can be spun out and generalized into a separate pip package

## License

The code is published under MIT License.

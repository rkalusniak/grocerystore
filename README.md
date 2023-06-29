# Grocery Store Model
==============================

This project uses the SimPy environment to model a basic grocery store. 
The entities for this model are customers. The model maps the customers’ trip through the grocery store, records key time stamps, and performs some 

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details **Not currently used**
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
	│	└──Visualize_Model.pptx	<-This the the PPT used to make the image for the notebook below
    │
    ├── notebooks          <- Jupyter notebooks and support. Naming convention is a number (for ordering),
    │   │                      the creator's initials, and a short `_` delimited description, e.g.
    │   │                      `1.0-jqp-initial-data-exploration`.
	│	│
	│	├── images			<- This includes all the images used in the notebook
	│	│	└──Visualize_Model.png <- This is a flow chart of the grocery store model
	│	│
	│	├── input			<- This includes any input files for  the notebook **Not currently used**
	│	├── output			<- This is the folder that holds the excel files when the model is run in the notebook
    │   └──1.0-rek-explaining-grocery-store.ipynb <- Main notebook
	│
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials. **Not currently used**
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc. **Not currently used**
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- Makes project pip installable (pip install -e .) so grocerystore can be imported
    ├── src                <- Source code for use in this project.
    │   ├── grocerystore <- Main package folder
    │   │   ├──__init__.py    <- Marks as package
    │   │   ├── grocerystore.py  <- This is the Python source code for the model
	│	│	├── input	<- This folder includes the config files for running the model 
    │	│	└── output	<- This folder includes all the output files from the model



--------

<p><small>Project based on a simplified version of the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

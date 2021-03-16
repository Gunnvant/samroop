# Samroop:

Samroop (समरूप) is python utility that can be used to locate duplicate pdf files on your file system. It uses classical ml to create an index and then finds out which files are most similar to a given file and also locate the paths of duplicates.

This code was created over a conversation with a friend who is a collector of ebooks but often ends up having many copies of the same book. 


## Usage:
You will only need to make changes in the `run.py` file. Change the `path_parent` to the path where you want to locate duplicates. This could be the path to the main folder where you keep your ebooks or the home directory of your computer.

This script will do the necessary indexation. Post which you can use the `query_logic.ipynb` to extract duplicate pdfs and their paths.

## Requirements:
If you have an anaconda environment, then the following python packages will have to be installed.
```
gensim
tika
```
`tika` will need java8, make sure the java path is properly set on your machine.
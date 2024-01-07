# modular_code_data_science
Samples project on how to write production level code for data scientist

Background:
I have seen a lot of my colleagues use a single jupyter notebook file for cleaning, statistical analysis, building model, cross validation, etc. This practice was bad because:
1. Your jupyter notebook might only run in your local environment, someone need to create a single python file to execute all your code if you want to deploy it in production or pipeline
2. When you want to do another analysis, you will copy-paste most of your code from another jupyter notebook because you use the same code for training model, cleaning, etc. A general rule of thumb, if you use the same code more than 2 times, then it's better to write a function (DRY principles)

This repo will give you examples on how to write production level code for Data Scientist, mainly modular/reusable and functional code in .py scripts, clean annotations, better variables names and classes.

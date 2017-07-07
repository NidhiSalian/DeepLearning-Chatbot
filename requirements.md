
# Requirements

  Below is a comprehensive list of the python libraries you will need to install before you can run this project.
  
     python 3.5
     
     tensorflow 1.1

     hdf5 1.8.18

     keras 2.0.2

     numpy 1.12.1

     scipy 0.19.1

     cython 0.25.2

     spacy 1.8.2

     nltk 3.2.4

     scikit-learn 0.18.2

A simple conda (preferably with conda-forge) or pip install will suffice for all the above, taking care of any prerequisites.

In fact, if you're creating a new conda environment, using the following commands should do all of that for you.

     conda config --add channels conda-forge
     conda create -n <new_env_name> python==3.6 cython nltk==3.2.4 keras spacy==1.8.2 tensorflow==1.1 scikit-learn 

Ypu can activate this new environment using:

    source activate <new_env_name>

And deactivate it using:

    source deactivate <new_env_name>

Windows Users help on activation and deactivation of conda environments - [here](https://stackoverflow.com/questions/20081338/how-to-activate-an-anaconda-environment).

You will need the complete English Language model for thorough and effective predictions.

So after you install spacy, run the following commands:

    python -m spacy download en_core_web_md
    python -m spacy link en_core_web_md en --force

That's it.
We're all set.

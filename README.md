# Random Fourier Features for PAC-Bayesian Domain Adaptation
This python code has been used to conduct the experiments
presented in Section 5 of the following report https://julienbastian.github.io/files/Domain%20Adaptation%20from%20a%20PAC-Bayesian%20Random%20Features%20perspective.pdf

It is an adaptation of the code (available here : https://github.com/gletarte/pbrff) that has been used to conduct the experiments
of the following paper:

> Gaël Letarte, Emilie Morvant, Pascal Germain.
> Pseudo-Bayesian Learning with Kernel Fourier Transform as Prior
http://proceedings.mlr.press/v89/letarte19a.html


## Important files
* ``experiment.py`` contains the code used to launch the toy experiments.
* ``experiment_real.py`` contains the code used to launch the experiments on the real datas.
* ``pbrff/landmarks_based.py`` implements the algorithm PBDA-landmarks.
* ``pbrff/landmarks_selector.py`` implements the landmark selection in the target domain.

## Launching an experiment
### Toy experiments
In order to launch the toy experimentq, launch ``experiment.py`` 
```zsh
python experiment.py
```
with the following arguments:
* **-d**, **--datasets** with a name to give to the results files.
* **-l**, **--landmarks-method** with either "random" or "clustering" to select the landmarks selection method for the _landmarks_based_ experiment.
* **-e**, **--experiments** with "landmarks_based".
* **-n**, **--n-cpu** with the desired number of cpus to be used or "-1" to use all available.
* **-r**, **--reversevalidation** with "yes" if reverse validation should be performed, "no" otherwise.

Here is an example:
```zsh
python experiment.py -d two_moons -e landmarks_based -l random -n -1
```

### Real data experiments
In order to launch the toy experimentq, launch ``experiment.py`` 
```zsh
python experiment.py
```
with the following arguments:
* **-dsource**, **--datasetsource** with the path to the file with the source data.
* **-dtarget**, **--datasettarget** with the path to the file with the target data.
* **-dtest**, **--datasettest** with the path to the file with the test data.
* **-l**, **--landmarks-method** with either "random" or "clustering" to select the landmarks selection method for the _landmarks_based_ experiment.
* **-e**, **--experiments** with "landmarks_based".
* **-n**, **--n-cpu** with the desired number of cpus to be used or "-1" to use all available.
* **-r**, **--reversevalidation** with "yes" if reverse validation should be performed, "no" otherwise.

Here is an example:
```zsh
python experiment.py -dsource ./data/books.dvd_source.svmlight -dtarget ./data/books.dvd_target.svmlight -dtest ./data/books.test.svmlight -e landmarks_based -l random -n -1
```

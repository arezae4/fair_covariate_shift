# fair_covariate_shift
This is the code for our paper [Robust Fairness Under Covariate Shift](https://arxiv.org/abs/2010.05166) published in AAAI 2021.


### Abstract 

Making predictions that are fair with regard to protected group membership (race, gender, age, etc.) has become an important requirement for classification algorithms. Existing techniques derive a fair model from sampled labeled data relying on the assumption that training and testing data are identically and independently drawn (iid) from the same distribution. In practice, distribution shift can and does occur between training and testing datasets as the characteristics of individuals interacting with the machine learning system change. We investigate fairness under covariate shift, a relaxation of the iid assumption in which the inputs or covariates change while the conditional label distribution remains the same. We seek fair decisions under these assumptions on target data with unknown labels. We propose an approach that obtains the predictor that is robust to the worst-case in terms of target performance while satisfying target fairness requirements and matching statistical properties of the source data. We demonstrate the benefits of our approach on benchmark prediction tasks. 

### Experiments

To run the experiment for each dataset run:

```console
$ python test_fair_covariate_shift.py --help
$ python test_fair_covariate_shift.py --dataset compas --repeat 10 --alpha 1.5 --beta 3
```

### Cite
```
@article{rezaei2020robust,
  title={Robust Fairness under Covariate Shift},
  author={Rezaei, Ashkan and Liu, Anqi and Memarrast, Omid and Ziebart, Brian},
  journal={arXiv preprint arXiv:2010.05166},
  year={2020}
}
```

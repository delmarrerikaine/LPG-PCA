# LPG-PCA-PY
Python Implementation of LPG-PCA algorithm [1]

Algorithm is in the lpg_pca_impl.py file. 
compare.py used for comparison of the obtained results with median filter, block-matching and 3D filtering (BM3D), non-local means filter algorithms. For BM3D please look at original repository.

In order to run the code, execute the next commands is the repository folder:

```
  pip install -r requirements.txt
  python compare.py
```

[1] Lei Zhang, Weisheng Dong, David Zhang, and Guangming Shi. Two-stage image denoising by principal component analysis with local pixel grouping. Pattern Recognition, 43(4):1531-1549, 2010.

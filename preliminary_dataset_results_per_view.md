# Preliminary results per view for each dataset

## Command and script employed
```python
>> python.exe ./dataset_validation.py
```

## Table with the performance of MiniBatch-kmeans

### Reuters (5 views)

|    View    |  F1 (macro)  |  Pr (macro)  |  Rec (macro)  |  NMI  |  Vm   |
|------------|--------------|--------------|---------------|-------|-------|
| english_K6 |    0.265     |    0.265     |     0.265     | 0.026 | 0.026 |
| france_K6  |    0.272     |    0.272     |     0.272     | 0.282 | 0.282 |
| german_K6  |    0.154     |    0.154     |     0.154     | 0.190 | 0.190 |
| italian_K6 |    0.266     |    0.266     |     0.266     | 0.111 | 0.111 |
| spanish_K6 |    0.066     |    0.066     |     0.066     | 0.198 | 0.198 |

### Handwritten (6 views)

|  View   |  F1 (macro)  |  Pr (macro)  |  Rec (macro)  |  NMI  |  Vm   |
|---------|--------------|--------------|---------------|-------|-------|
| pix_K10 |    0.082     |    0.082     |     0.082     | 0.690 | 0.690 |
| fou_K10 |    0.083     |    0.083     |     0.083     | 0.620 | 0.620 |
| fac_K10 |    0.046     |    0.046     |     0.046     | 0.639 | 0.639 |
| zer_K10 |    0.095     |    0.095     |     0.095     | 0.494 | 0.494 |
| kar_K10 |    0.092     |    0.092     |     0.092     | 0.581 | 0.581 |
| mor_K10 |    0.106     |    0.106     |     0.106     | 0.478 | 0.478 |

### Caltech-7 (6 views)

|    View     |  F1 (macro)  |  Pr (macro)  |  Rec (macro)  |  NMI  |  Vm   |
|-------------|--------------|--------------|---------------|-------|-------|
|  gabor_K7   |    0.154     |    0.154     |     0.154     | 0.140 | 0.140 |
|    wm_K7    |    0.035     |    0.035     |     0.035     | 0.298 | 0.298 |
| centrist_K7 |    0.014     |    0.014     |     0.014     | 0.378 | 0.378 |
|   hog_K7    |    0.387     |    0.387     |     0.387     | 0.608 | 0.608 |
|   gist_K7   |    0.225     |    0.225     |     0.225     | 0.516 | 0.516 |
|   lbp_K7    |    0.033     |    0.033     |     0.033     | 0.398 | 0.398 |

### Caltech-20 (6 views)

|     View     |  F1 (macro)  |  Pr (macro)  |  Rec (macro)  |  NMI  |  Vm   |
|--------------|--------------|--------------|---------------|-------|-------|
|  gabor_K20   |    0.027     |    0.027     |     0.027     | 0.263 | 0.263 |
|    wm_K20    |    0.034     |    0.034     |     0.034     | 0.371 | 0.371 |
| centrist_K20 |    0.092     |    0.092     |     0.092     | 0.358 | 0.358 |
|   hog_K20    |    0.012     |    0.012     |     0.012     | 0.610 | 0.610 |
|   gist_K20   |    0.094     |    0.094     |     0.094     | 0.572 | 0.572 |
|   lbp_K20    |    0.007     |    0.007     |     0.007     | 0.488 | 0.488 |

### Nus-Wide (5 views)

|   View   |  F1 (macro)  |  Pr (macro)  |  Rec (macro)  |  NMI  |  Vm   |
|----------|--------------|--------------|---------------|-------|-------|
|  ch_K31  |    0.026     |    0.026     |     0.026     | 0.089 | 0.089 |
|  cm_K31  |    0.038     |    0.038     |     0.038     | 0.105 | 0.105 |
| corr_K31 |    0.035     |    0.035     |     0.035     | 0.100 | 0.100 |
| edh_K31  |    0.023     |    0.023     |     0.023     | 0.121 | 0.121 |
|  wt_K31  |    0.037     |    0.037     |     0.037     | 0.102 | 0.102 |

### BBC-2 (2 views)

|  View  |  F1 (macro)  |  Pr (macro)  |  Rec (macro)  |  NMI  |  Vm   |
|--------|--------------|--------------|---------------|-------|-------|
| v0_K5  |    0.179     |    0.179     |     0.179     | 0.005 | 0.005 |
| v1_K5  |    0.127     |    0.127     |     0.127     | 0.097 | 0.097 |

### BBC-3 (3 views)

|  View  |  F1 (macro)  |  Pr (macro)  |  Rec (macro)  |  NMI  |  Vm   |
|--------|--------------|--------------|---------------|-------|-------|
| v0_K5  |    0.159     |    0.159     |     0.159     | 0.127 | 0.127 |
| v1_K5  |    0.114     |    0.114     |     0.114     | 0.164 | 0.164 |
| v2_K5  |    0.256     |    0.256     |     0.256     | 0.007 | 0.007 |

### BBC-4 (4 views)

|  View  |  F1 (macro)  |  Pr (macro)  |  Rec (macro)  |  NMI  |  Vm   |
|--------|--------------|--------------|---------------|-------|-------|
| v0_K5  |    0.115     |    0.115     |     0.115     | 0.015 | 0.015 |
| v1_K5  |    0.190     |    0.190     |     0.190     | 0.065 | 0.065 |
| v2_K5  |    0.156     |    0.156     |     0.156     | 0.089 | 0.089 |
| v3_K5  |    0.118     |    0.118     |     0.118     | 0.022 | 0.022 |

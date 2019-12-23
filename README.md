# [MULE: Multimodal Universal Language Embedding (MULE 2020 Oral)](https://arxiv.org/pdf/1909.03493.pdf)

![](docs/fig1.png)

## Training and Testing
  ```Shell
  ./run_mule.sh [MODE] [GPU_ID] [DATASET] [TAG] [EPOCH]
  # MODE {train, test, val} which indicates if you want to train the model or evaluate it using test or val splits
  # GPU_ID is the GPU you want to test on
  # DATASET {multi30k, coco} is defined in run_mule.sh
  # TAG is an experiment name
  # EPOCH optional, epoch number to test, if not provided, best model on validation data is used
  # Examples:
  ./run_mule.sh train 0 multi30k mule
  ./run_mule.sh train 1 coco mule
  ./run_mule.sh test 1 coco mule
  ./run_mule.sh val 0 multi30k mule 20
  ```
  
By default, trained networks are saved under:

```
models/[NET]/[DATASET]/{TAG}/
```

### Citation
If you find our code useful please consider citing:

    @inproceedings{kimMULEAAAI2020,
      title={{MULE: Multimodal Universal Language Embedding}},
      author={Donghyun Kim and Kuniaki Saito and Kate Saenko and Stan Sclaroff and Bryan A Plummer},
      booktitle={AAAI Conference on Artificial Intelligence},
      year={2020}
    }

## Fine-grained fact-checking in political statements

### Simple baseline

- Run simple baseline

  - The majority label, and accuracy, F1 score on the validation and test set will be shown by running

    ```
    python simple_basline.py
    ```

### Strong baseline: CNN

- Run CNN model

  - The best result on the validation and test dataset can be reproduced by putting the trained model in the data folder as the script and running

    ```
    python strong_baseline.py --eval --test 
    ```

  - To train the CNN from scratch, run

    ```
    python strong_baseline.py --train
    ```

    The best model will be saved in the data folder.

### Extension 1

In this extension, we change the dataset from Liar to Liar-Plus, and change model structure from a CNN to BiLSTM + Attention

- Run the BiLSTM model

  - The best result on the validation and test dataset can be reproduced by putting the trained model in the data folder as the script and running

    ```
    python ex1_main.py --eval --test --save_testy
    ```

    The flag `save_testy` here saves true label and predicted label for the test set in the output folder, which will be used in the evaluation script.

  - To train the CNN from scratch, run

    ```
    python ex1_main.py --train
    ```

    The best model will be saved in the data folder.

### Extension 2

In this extension, we propose a transfer learning based approach using pretrained XLNet model.

- Required package

```
transformers                       4.13.0
sentencepiece                      0.1.96
gensim                             4.0.1
torch                              1.10.0
```

- Download Pretrained Model

  - Run the download scipt to get the xlnet-base-cased model from Huggingface

    ```
    ./download.sh
    ```


- Run the XLNet model

  - The best result on the validation and test dataset can be reproduced by putting the trained model in the same folder as the data and running

    ```
    python ex2_main.py --eval --test 
    ```


  - To tune the XLNet, run

    ```
    python ex2_main.py --train
    ```

    

### Evaluation script

- Run the evaluation script

  - Make sure the true label and predicted label for the test set mentioned before is put in the output folder, and run 

    ```
    python evaluation.py
    ```

    This will print the accuracy and F1 score, and show the confusion matrix. Examples are shown in `output/README.md`


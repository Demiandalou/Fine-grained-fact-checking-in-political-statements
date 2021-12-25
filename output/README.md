### Evaluation 

- Run the evaluation script

  - Make sure the true label and predicted label for the test set mentioned before is put in the output folder, and run 

    ```
    python ../code/evaluation.py
    ```

    This will print the accuracy and F1 score, and show the confusion matrix. 

- Examples

  - For the BiLSTM+Attention model in the extension 1, the result on the test set is 

    ```
    acc:  0.27624309392265195 Micro f1 0.23388237740515952
    ```

    The confusion matrix is ![Figure_1.png](https://s2.loli.net/2021/12/16/zL8SXOlbfWaNpyP.png)


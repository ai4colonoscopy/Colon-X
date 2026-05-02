# Troubleshooting Guide — Common Issues and Solutions

## Q1. Issues with `flash-attn` Version Compatibility

- **Problem:** Errors occur during the installation of the `flash-attn` package.
- **Cause:** The `flash-attn` package has strict version requirements for PyTorch and CUDA. Incompatible versions can lead to installation failures.
- **Solution:** Ensure that you are using compatible versions of PyTorch and CUDA. Our setup uses `flash-attn 2.7.1.post4` with PyTorch 2.6.0 and CUDA 11.8. You can install `flash-attn` via the offline wheel file ([Google Drive](https://drive.google.com/file/d/1XEUqJqDxXpWsL4W5sFmJjCY0YxJRyGKl/view?usp=sharing)), and then run:

    ```shell
    # If the installation fails, run the following command to install flash-attn
    pip install cache/flash-attn/flash_attn-2.7.1.post4+cu11torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
    ```

## Q2. Compatibility of various ulcerative colitis labels 
- **Problem:**  
  There may appear to be overlap between the general `CLS#33` ulcerative colitis label and fine-grained labels (`CLS#34`~`CLS#40`) for ulcerative colitis grading.
- **Explanation:**
The `CLS#33` ulcerative colitis label is used only for the classification task, whereas the fine-grained ulcerative colitis grading labels (`CLS#34`~`CLS#40`) are used only for the ulcerative colitis grading task. These labels are separated at the task-design level, which avoids direct label conflicts.

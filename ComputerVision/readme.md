

### install dependencies
pip install -r requirements.txt
<!-- pip install -r ./ComputerVision/requirements.txt -->


### Run demos:
streamlit run ./Demos/img_proc/img_proc.py



```

elif filter_option == "OpenPose":

device = "cuda:0" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

dwpose = DWposeDetector(device=device)



skeleton = dwpose(orig_image.convert("RGB"), output_type="pil", include_hands=True, include_face=True)

st.image(skeleton, caption="OpenPose Skeleton", use_column_width=True)
```


# Drive link to the course materials: 
https://drive.google.com/drive/folders/1QHDrDbT7QWLw92_rkswI_gqo02tVNjb9?usp=sharing
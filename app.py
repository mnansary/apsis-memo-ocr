
import os

import matplotlib.pyplot as plt
import streamlit as st

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import tensorflow as tf

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if any(gpu_devices):
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)

from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from doctr.utils.visualization import visualize_page

#DET_ARCHS = ["db_resnet50", "db_mobilenet_v3_large"]
#RECO_ARCHS = ["crnn_vgg16_bn", "crnn_mobilenet_v3_small", "master", "sar_resnet31"]


def main():

    # Wide mode
    st.set_page_config(layout="wide")

    # Designing the interface
    st.title("APSIS Doc OCR Module: Document Text Recognition(beta)")
    # For newline
    st.write('\n')
    # Instructions
    st.markdown("*Hint: click on the top-right corner of an image to enlarge it!*")
    # Set the columns
    cols = st.columns((1, 1, 1))
    cols[0].subheader("Input page")
    cols[1].subheader("Segmentation heatmap")
    cols[2].subheader("OCR output")
    
    # Sidebar
    # File selection
    st.sidebar.title("Document selection")
    # Disabling warning
    st.set_option('deprecation.showfileUploaderEncoding', False)
    # Choose your own image
    uploaded_file = st.sidebar.file_uploader("Upload files", type=['png', 'jpeg', 'jpg'])
    if uploaded_file is not None:
        doc = DocumentFile.from_images(uploaded_file.read())
        page_idx = st.sidebar.selectbox("Page selection", [idx + 1 for idx in range(len(doc))]) - 1
        cols[0].image(doc[page_idx])

    det_arch = "db_mobilenet_v3_large"
    reco_arch = "sar_resnet31"

    # For newline
    st.sidebar.write('\n')

    if st.sidebar.button("Analyze page"):

        if uploaded_file is None:
            st.sidebar.write("Please upload a document")

        else:
            with st.spinner('Loading model...'):
                predictor = ocr_predictor(det_arch, reco_arch, pretrained=True)

            with st.spinner('Analyzing...'):

                # Forward the image to the model
                processed_batches = predictor.det_predictor.pre_processor([doc[page_idx]])
                out = predictor.det_predictor.model(processed_batches[0], return_model_output=True)
                seg_map = out["out_map"]
                seg_map = tf.squeeze(seg_map[0, ...], axis=[2])
                seg_map = cv2.resize(seg_map.numpy(), (doc[page_idx].shape[1], doc[page_idx].shape[0]),
                                     interpolation=cv2.INTER_LINEAR)
                # Plot the raw heatmap
                fig, ax = plt.subplots()
                ax.imshow(seg_map)
                ax.axis('off')
                cols[1].pyplot(fig)

                # Plot OCR output
                out = predictor([doc[page_idx]])
                fig = visualize_page(out.pages[0].export(), doc[page_idx], interactive=False)
                cols[2].pyplot(fig)

                
if __name__ == '__main__':
    main()

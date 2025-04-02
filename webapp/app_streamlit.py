from predict import *

def webapp_fn():
    st.title("Object Detection Web App")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    model_name = st.selectbox("Choose a model", ["ssd", "retina", "faster_rcnn", "fcos"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_container_width=True)

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)

        st.download_button(
            label="Download Image",
            data=buf,
            file_name="uploaded_image.png",
            mime="image/png"
        )
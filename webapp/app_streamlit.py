from predict import *

def webapp_fn():
    st.title("Object Detection Web App")
    st.markdown("[Visit my portfolio](https://amir-hofo.github.io/Portfolio/)")

    uploaded_file= st.file_uploader("Choose an image...", type= ["jpg", "jpeg", "png"])
    model_name= st.selectbox("Choose a model", ["ssd", "retina", "faster_rcnn", "fcos"])

    if uploaded_file:
        image= Image.open(uploaded_file)
        model= Model(model_name)
        model.eval()
        image= prediction_fn(image, model)

        # image= Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        st.image(image, caption= "Done.", use_container_width= True)

        buf= io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)

        st.download_button(
            label="Download Image",
            data=buf,
            file_name="OD_image.png",
            mime="image/png"
        )
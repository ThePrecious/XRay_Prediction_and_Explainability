"""Produces gradio app"""
import os
from pathlib import Path

import gradio as gr

from .app_utils import predict, explain_gradient, explain_input_x_gradient, explain_integrated_gradients, explain_gifsplanation

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # do not use GPU

APP_DIR = Path(__file__).resolve().parent  # application directory

DEFAULT_PORT = 11700


def main():
    
    frontend = make_frontend()
    frontend.launch(
        server_name="0.0.0.0",  # make server accessible, binding all interfaces  # noqa: S104
        server_port=DEFAULT_PORT,  # set a port to bind to, failing if unavailable
        share=True,  # should we create a (temporary) public link on https://gradio.app?
    )


def make_frontend():
    """Creates a gradio.Interface frontend for an image to text function."""

    model_choices = ["densenet121-res224-all", "densenet121-res224-rsna","densenet121-res224-nih","densenet121-res224-pc","densenet121-res224-chex","densenet121-res224-mimic_nb","densenet121-res224-mimic_ch"]

    target_choices = ['Atelectasis','Consolidation','Infiltration','Pneumothorax','Edema','Emphysema',
                    'Fibrosis','Effusion','Pneumonia','Pleural_Thickening','Cardiomegaly','Nodule','Mass',
                    'Hernia','Lung Lesion','Fracture','Lung Opacity','Enlarged Cardiomediastinum']

    headers = ['Pathology', 'Score','Diagnosis']

    examples_dir = Path("src") / "examples"
    example_fnames = [elem for elem in os.listdir(examples_dir) if elem.endswith(".png")]
    example_paths = [examples_dir / fname for fname in example_fnames]
    examples = [[str(path)] for path in example_paths]


    # build a basic browser interface to a Python function
    frontend = gr.Blocks()

    with frontend:
        gr.Markdown(
            """
            # X-Ray Diagnosis AI Assistant
            Human-in-the-loop interface for exploring TorchXRayVision models and data.
            """
            )
        #layout
        with gr.Tab("Prediction"):
            with gr.Row():
                with gr.Column():
                    with gr.Tab("Input"):
                        input_image = gr.Image(label="X-ray image")
                        select_model = gr.Dropdown(label="Select model", choices=model_choices)
                        with gr.Row():
                            submit_button = gr.Button("Submit")
                        gr.Examples(examples, inputs=input_image)
                with gr.Column():
                    with gr.Tab("Report"):
                        report = gr.Dataframe(headers=headers)

        submit_button.click(predict, [input_image, select_model], report) 

        with gr.Tab("Explanation"):
            with gr.Row():
                with gr.Column():
                    select_target = gr.Dropdown(label="Select target", choices=target_choices)
                    with gr.Row(equal_height=True):
                        with gr.Column():
                            with gr.Tab("Im..."):
                                original_image = gr.Image(label='Original', interactive=False)
                        with gr.Column():
                            with gr.Tab("Gra..."):
                                gradient_plot = gr.Plot(label="Gradient")
                            with gr.Tab("XGr..."):
                                input_x_gradient_plot = gr.Plot(label='InputXGradient')
                            with gr.Tab("Int..."):
                                integrated_gradients_plot = gr.Plot(label="IntegratedGradients")
                            with gr.Tab("Gif..."):
                                gifsplanation_vid = gr.Video(label="Gifsplanation")
                
        input_image.change(lambda s: s, inputs=input_image, outputs=original_image)
        select_target.change(explain_gradient, inputs=[input_image,select_model, select_target], outputs=gradient_plot)
        select_target.change(explain_input_x_gradient, inputs=[input_image,select_model, select_target], outputs=input_x_gradient_plot)
        select_target.change(explain_integrated_gradients, inputs=[input_image,select_model, select_target], outputs=integrated_gradients_plot)
        select_target.change(explain_gifsplanation, inputs=[input_image,select_model, select_target], outputs=gifsplanation_vid)

    return frontend
    
if __name__ == "__main__":
    main()


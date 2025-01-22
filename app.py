import gradio as gr
from consistory_CLI import run_batch, run_cached_anchors
import os

def process_inputs(run_type, gpu, seed, mask_dropout, same_latent, style, subject, 
                  concept_tokens, settings_text, cache_cpu_offloading, out_dir):
    # Convert string inputs to appropriate types
    concept_tokens = [token.strip() for token in concept_tokens.split(',')] if concept_tokens else ["dog"]
    
    # Split settings by newlines and filter out empty lines
    settings = [setting.strip() for setting in settings_text.split('\n') if setting.strip()]
    
    # Create output directory if specified
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    # Run appropriate function based on run_type
    if run_type == "batch":
        images, image_all = run_batch(
            gpu=gpu,
            seed=seed,
            mask_dropout=mask_dropout,
            same_latent=same_latent,
            style=style,
            subject=subject,
            concept_token=concept_tokens,
            settings=settings,
            out_dir=out_dir
        )
    else:  # cached
        images, image_all = run_cached_anchors(
            gpu=gpu,
            seed=seed,
            mask_dropout=mask_dropout,
            same_latent=same_latent,
            style=style,
            subject=subject,
            concept_token=concept_tokens,
            settings=settings,
            cache_cpu_offloading=cache_cpu_offloading,
            out_dir=out_dir
        )
    
    # Return both individual images and the combined image
    return images + [image_all] if image_all is not None else images

# Create the Gradio interface
with gr.Blocks(title="Consistory Image Generator") as demo:
    gr.Markdown("## Consistory Image Generator")
    
    with gr.Row():
        with gr.Column():
            # Basic Settings
            run_type = gr.Radio(
                choices=["batch", "cached"],
                value="batch",
                label="Run Type"
            )
            
            gpu = gr.Number(
                value=0,
                label="GPU ID",
                precision=0
            )
            
            seed = gr.Number(
                value=40,
                label="Seed",
                precision=0
            )
            
            mask_dropout = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.5,
                label="Mask Dropout"
            )
            
            same_latent = gr.Checkbox(
                value=False,
                label="Use Same Latent"
            )
            
        with gr.Column():
            # Generation Settings
            style = gr.Textbox(
                value="A photo of ",
                label="Style Prompt"
            )
            
            subject = gr.Textbox(
                value="a cute dog",
                label="Subject"
            )
            
            concept_tokens = gr.Textbox(
                value="dog",
                label="Concept Tokens (comma-separated)"
            )
            
            settings = gr.Textbox(
                value="sitting in the beach\nin the circus",
                label="Settings (one per line)",
                lines=5,
                placeholder="Enter each setting on a new line:\nsitting in the beach\nin the circus\nstanding in the snow"
            )
            
            cache_cpu_offloading = gr.Checkbox(
                value=False,
                label="Cache CPU Offloading",
                visible=False  # Only show when run_type is "cached"
            )
            
            out_dir = gr.Textbox(
                value="out",
                label="Output Directory"
            )

    # Show/hide cache_cpu_offloading based on run_type
    def update_cache_visibility(run_type):
        return {"visible": run_type == "cached"}
    
    run_type.change(
        fn=update_cache_visibility,
        inputs=[run_type],
        outputs=[cache_cpu_offloading]
    )
    
    # Submit button
    submit_btn = gr.Button("Generate Images")
    
    # Output section
    with gr.Row():
        # Gallery for individual images
        gallery = gr.Gallery(
            label="Generated Images",
            show_label=True,
            elem_id="gallery",
            columns=3,
            height="auto"
        )
    
    # Connect the submit button to the processing function
    submit_btn.click(
        fn=process_inputs,
        inputs=[
            run_type, gpu, seed, mask_dropout, same_latent,
            style, subject, concept_tokens, settings,
            cache_cpu_offloading, out_dir
        ],
        outputs=[gallery]
    )

if __name__ == "__main__":
    demo.launch()
import os
import numpy as np
import torch
import gradio as gr  
import spaces
from typing import Optional, Tuple
from funasr import AutoModel
from pathlib import Path
os.environ["TOKENIZERS_PARALLELISM"] = "false"
if os.environ.get("HF_REPO_ID", "").strip() == "":
    os.environ["HF_REPO_ID"] = "openbmb/VoxCPM-0.5B"

import voxcpm


class VoxCPMDemo:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Running on device: {self.device}")

        # ASR model for prompt text recognition
        self.asr_model_id = "iic/SenseVoiceSmall"
        self.asr_model: Optional[AutoModel] = AutoModel(
            model=self.asr_model_id,
            disable_update=True,
            log_level='DEBUG',
            device="cuda:0" if self.device == "cuda" else "cpu",
        )

        # TTS model (lazy init)
        self.voxcpm_model: Optional[voxcpm.VoxCPM] = None
        self.default_local_model_dir = "./models/VoxCPM-0.5B"

    # ---------- Model helpers ----------
    def _resolve_model_dir(self) -> str:
        """
        Resolve model directory:
        1) Use local checkpoint directory if exists
        2) If HF_REPO_ID env is set, download into models/{repo}
        3) Fallback to 'models'
        """
        if os.path.isdir(self.default_local_model_dir):
            return self.default_local_model_dir

        repo_id = os.environ.get("HF_REPO_ID", "").strip()
        if len(repo_id) > 0:
            target_dir = os.path.join("models", repo_id.replace("/", "__"))
            if not os.path.isdir(target_dir):
                try:
                    from huggingface_hub import snapshot_download  # type: ignore
                    os.makedirs(target_dir, exist_ok=True)
                    print(f"Downloading model from HF repo '{repo_id}' to '{target_dir}' ...")
                    snapshot_download(repo_id=repo_id, local_dir=target_dir, local_dir_use_symlinks=False)
                except Exception as e:
                    print(f"Warning: HF download failed: {e}. Falling back to 'data'.")
                    return "models"
            return target_dir
        return "models"

    def get_or_load_voxcpm(self) -> voxcpm.VoxCPM:
        if self.voxcpm_model is not None:
            return self.voxcpm_model
        print("Model not loaded, initializing...")
        model_dir = self._resolve_model_dir()
        print(f"Using model dir: {model_dir}")
        self.voxcpm_model = voxcpm.VoxCPM(voxcpm_model_path=model_dir)
        print("Model loaded successfully.")
        return self.voxcpm_model

    # ---------- Functional endpoints ----------
    def prompt_wav_recognition(self, prompt_wav: Optional[str]) -> str:
        if prompt_wav is None:
            return ""
        res = self.asr_model.generate(input=prompt_wav, language="auto", use_itn=True)
        text = res[0]["text"].split('|>')[-1]
        return text

    def generate_tts_audio(
        self,
        text_input: str,
        prompt_wav_path_input: Optional[str] = None,
        prompt_text_input: Optional[str] = None,
        cfg_value_input: float = 2.0,
        inference_timesteps_input: int = 10,
        do_normalize: bool = True,
        denoise: bool = True,
    ) -> Tuple[int, np.ndarray]:
        """
        Generate speech from text using QLX; optional reference audio for voice style guidance.
        Returns (sample_rate, waveform_numpy)
        """
        current_model = self.get_or_load_voxcpm()

        text = (text_input or "").strip()
        if len(text) == 0:
            raise ValueError("Please input text to synthesize.")

        prompt_wav_path = prompt_wav_path_input if prompt_wav_path_input else None
        prompt_text = prompt_text_input if prompt_text_input else None

        print(f"Generating audio for text: '{text[:60]}...'")
        wav = current_model.generate(
            text=text,
            prompt_text=prompt_text,
            prompt_wav_path=prompt_wav_path,
            cfg_value=float(cfg_value_input),
            inference_timesteps=int(inference_timesteps_input),
            normalize=do_normalize,
            denoise=denoise,
        )
        return (16000, wav)


# ---------- UI Builders ----------

def create_demo_interface(demo: VoxCPMDemo):
    """Build the Gradio UI for QLX demo."""
    # static assets (logo path)
    gr.set_static_paths(paths=[Path.cwd().absolute()/"assets"])

    with gr.Blocks(
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="gray",
            neutral_hue="slate",
            font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"]
        ),
        css="""
        .logo-container {
            text-align: center;
            margin: 0.5rem 0 1rem 0;
        }
        .logo-container img {
            height: 80px;
            width: auto;
            max-width: 200px;
            display: inline-block;
        }
        /* Bold accordion labels */
        #acc_quick details > summary,
        #acc_tips details > summary {
            font-weight: 600 !important;
            font-size: 1.1em !important;
        }
        /* Bold labels for specific checkboxes */
        #chk_denoise label,
        #chk_denoise span,
        #chk_normalize label,
        #chk_normalize span {
            font-weight: 600;
        }
        """
    ) as interface:
        # Header logo
        gr.HTML('<div class="logo-container"><img src="/gradio_api/file=assets/qlx_logo.png" alt="VoxCPM Logo"></div>')

        # Quick Start
        with gr.Accordion("📋 Quick Start Guide", open=False, elem_id="acc_quick"):
            gr.Markdown("""
            ### How to Use
            1. **(Optional) Provide a Voice Prompt** - Upload or record an audio clip to provide the desired voice characteristics for synthesis.  
             
            2. **(Optional) Enter prompt text** - If you provided a voice prompt, enter the corresponding transcript here (auto-recognition available).  
              
            3. **Enter target text** - Type the text you want the model to speak.  
            
            4. **Generate Speech** - Click the "Generate" button to create your audio.  
              
            """)

        # Pro Tips
        with gr.Accordion("💡 Pro Tips", open=False, elem_id="acc_tips"):
            gr.Markdown("""
            ### Prompt Speech Enhancement
            - **Enable** to remove background noise for a clean, studio-like voice, with an external ZipEnhancer component.  
              
            - **Disable** to preserve the original audio's background atmosphere.  
              

            ### Text Normalization
            - **Enable** to process general text with an external WeTextProcessing component.  
              
            - **Disable** to use QLX's native text understanding ability. For example, it supports phonemes input ({HH AH0 L OW1}), try it!  

            ### CFG Value
            - **Lower CFG** if the voice prompt sounds strained or expressive.  
            
            - **Higher CFG** for better adherence to the prompt speech style or input text.  
             

            ### Inference Timesteps
            - **Lower** for faster synthesis speed.  
            
            - **Higher** for better synthesis quality.  
            """)

        # Main controls
        with gr.Row():
            with gr.Column():
                prompt_wav = gr.Audio(
                    sources=["upload", 'microphone'],
                    type="filepath",
                    label="Prompt Speech (Optional, or let QLX improvise)",
                    value="./examples/example.wav",
                )
                DoDenoisePromptAudio = gr.Checkbox(
                    value=False,
                    label="Prompt Speech Enhancement",
                    elem_id="chk_denoise",
                    info="We use ZipEnhancer model to denoise the prompt audio."
                )
                with gr.Row():
                    prompt_text = gr.Textbox(
                        value="Just by listening a few minutes a day, you'll be able to eliminate negative thoughts by conditioning your mind to be more positive.",
                        label="Prompt Text",
                        placeholder="Please enter the prompt text. Automatic recognition is supported, and you can correct the results yourself..."
                    )
                run_btn = gr.Button("Generate Speech", variant="primary")

            with gr.Column():
                cfg_value = gr.Slider(
                    minimum=1.0,
                    maximum=3.0,
                    value=2.0,
                    step=0.1,
                    label="CFG Value (Guidance Scale)",
                    info="Higher values increase adherence to prompt, lower values allow more creativity"
                )
                inference_timesteps = gr.Slider(
                    minimum=4,
                    maximum=30,
                    value=10,
                    step=1,
                    label="Inference Timesteps",
                    info="Number of inference timesteps for generation (higher values may improve quality but slower)"
                )
                with gr.Row():
                    text = gr.Textbox(
                        value="Qualex is a company that brings insights in your data.",
                        label="Target Text",
                    )
                with gr.Row():
                    DoNormalizeText = gr.Checkbox(
                        value=False,
                        label="Text Normalization",
                        elem_id="chk_normalize",
                        info="We use wetext library to normalize the input text."
                    )
                audio_output = gr.Audio(label="Output Audio")

        # Wiring
        run_btn.click(
            fn=demo.generate_tts_audio,
            inputs=[text, prompt_wav, prompt_text, cfg_value, inference_timesteps, DoNormalizeText, DoDenoisePromptAudio],
            outputs=[audio_output],
            show_progress=True,
            api_name="generate",
        )
        prompt_wav.change(fn=demo.prompt_wav_recognition, inputs=[prompt_wav], outputs=[prompt_text])

    return interface


def run_demo(server_name: str = "localhost", server_port: int = 7860, show_error: bool = True):
    demo = VoxCPMDemo()
    interface = create_demo_interface(demo)
    # Recommended to enable queue on Spaces for better throughput
    interface.queue(max_size=10).launch(server_name=server_name, server_port=server_port, show_error=show_error)


if __name__ == "__main__":
    run_demo()
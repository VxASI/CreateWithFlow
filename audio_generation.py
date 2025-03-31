import os
import mlx.core as mx
from mlx_audio.tts.generate import generate_audio
from typing import Optional, Union, List

def generate_speech(
    text: str,
    model_path: str = "prince-canuma/Kokoro-82M",
    voice: str = "af_heart",
    speed: float = 1.0,
    output_dir: str = "generated_audio",
    file_prefix: str = "speech",
    audio_format: str = "wav",
    sample_rate: int = 24000,
    lang_code: Optional[str] = "a",  # Kokoro: (a)f_heart, or None for auto
    join_audio: bool = True,
    verbose: bool = True
) -> str:
    """
    Generate speech from text using MLX Audio TTS.
    
    Args:
        text (str): The text to convert to speech
        model_path (str): Path to the TTS model or HuggingFace model ID
        voice (str): Voice ID to use for generation
        speed (float): Speech speed multiplier (0.5 to 2.0)
        output_dir (str): Directory to save generated audio files
        file_prefix (str): Prefix for output audio files
        audio_format (str): Output audio format ('wav' or 'mp3')
        sample_rate (int): Audio sample rate in Hz
        lang_code (Optional[str]): Language code for multi-lingual models
        join_audio (bool): Whether to join multiple audio segments
        verbose (bool): Whether to print progress messages
    
    Returns:
        str: Path to the generated audio file
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Ensure file prefix doesn't contain path separators
        file_prefix = os.path.basename(file_prefix)
        
        # Construct the expected output path
        expected_output_path = os.path.join(output_dir, f"{file_prefix}.{audio_format}")
        
        # Generate the audio
        _ = generate_audio(
            text=text,
            model_path=model_path,
            voice=voice,
            speed=speed,
            lang_code=lang_code,
            file_prefix=os.path.join(output_dir, file_prefix),
            audio_format=audio_format,
            sample_rate=sample_rate,
            join_audio=join_audio,
            verbose=verbose
        )
        
        # Check if the file was generated
        if os.path.exists(expected_output_path):
            if verbose:
                print(f"Audio generated successfully: {expected_output_path}")
            return expected_output_path
        
        # If the expected path doesn't exist, try to find the file in the output directory
        files = os.listdir(output_dir)
        matching_files = [f for f in files if f.startswith(file_prefix) and f.endswith(f".{audio_format}")]
        
        if matching_files:
            actual_path = os.path.join(output_dir, matching_files[0])
            if verbose:
                print(f"Audio generated with different name: {actual_path}")
            return actual_path
            
        raise FileNotFoundError(f"Audio generation failed: output file not found in {output_dir}")
        
    except Exception as e:
        print(f"Error generating audio: {str(e)}")
        print(f"Expected output path would have been: {expected_output_path}")
        raise  # Re-raise the exception to be handled by the caller

if __name__ == "__main__":
    # Example usage with error handling
    try:
        text = """
        Welcome to the world of AI-generated speech!
        This is a demonstration of MLX Audio's text-to-speech capabilities.
        The audio quality is quite impressive, and it supports multiple languages and voices.
        """
        
        output_file = generate_speech(
            text=text,
            speed=1.2,
            file_prefix="demo_speech",
            verbose=True
        )
        
        print(f"Generated audio file: {output_file}")
    except Exception as e:
        print(f"Failed to generate speech: {e}") 